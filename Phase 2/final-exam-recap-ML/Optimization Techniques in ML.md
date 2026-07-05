- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Optimization Techniques in ML — Complete Question Bank|Optimization Techniques in ML — Complete Question Bank]]
- [[#SECTION 1: Problem Formulation & Why We Need Optimization|SECTION 1: Problem Formulation & Why We Need Optimization]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: Gradient Descent Foundations|SECTION 2: Gradient Descent Foundations]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: Types of Gradient Descent|SECTION 3: Types of Gradient Descent]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: Second-Order Methods|SECTION 4: Second-Order Methods]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: Adaptive Optimizers|SECTION 5: Adaptive Optimizers]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: Non-Convex Optimization|SECTION 6: Non-Convex Optimization]]
		- [[#Optimization Techniques in ML — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 6: Non-Convex Optimization#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 6: Non-Convex Optimization#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 6: Non-Convex Optimization#Top 7 Most Likely Exam Questions From This Topic|Top 7 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Optimization Techniques in ML — Complete Question Bank

---

# SECTION 1: Problem Formulation & Why We Need Optimization

---

**Question 1**
The optimization problem in machine learning is fundamental to all model training.

1. Formally define the optimization problem in ML. What do we have, what is our goal, and how do we achieve it? Explain the role of the loss function L(θ, x, y).
2. Explain why linear regression with squared error loss is one of the rare cases that admits a closed-form solution. What three properties of the loss function make this possible?
3. List four models that do NOT admit closed-form solutions. For each, explain specifically why setting ∇L(θ) = 0 fails to give an explicit solution.

### ✅ Answer

**Part 1:**
**Formal optimization problem in ML:**

**What we have:**
- Training dataset of N observations: {xₙ, yₙ} for n = 1,...,N
- A model f(x; θ) parametrized by θ (weights, biases, coefficients)

**What is our goal:**
Learn parameters θ such that predictions f(xᵢ; θ) are as close as possible to labels yᵢ — i.e., find:
θ* = argmin_θ L(θ, x, y)

**How we achieve it:**
1. Choose a loss function L that measures how wrong predictions are
2. Minimize L with respect to θ using either closed-form solutions (rarely) or iterative methods (usually)

**Role of L(θ, x, y):**
The loss function is the mathematical bridge between "how wrong is the model" and "which direction to adjust parameters." It:
- Maps all N prediction errors to a single scalar (the "cost")
- Is differentiable (usually) so we can compute the gradient direction
- Captures domain-specific error preferences (MSE penalizes large errors quadratically; cross-entropy penalizes confident wrong predictions exponentially)

The choice of L is not arbitrary — it encodes the statistical assumption about the data generating process. MSE corresponds to Gaussian noise assumption; cross-entropy corresponds to Bernoulli assumption.

**Part 2:**
**Three properties making linear regression closed-form:**

Linear regression with squared loss: L(w) = (1/N)||y - Xw||²

**Property 1 — Convexity:**
L(w) is a quadratic function of w — it forms a paraboloid in parameter space with a single lowest point. There are no local minima, no saddle points. Any critical point is the global minimum.

**Property 2 — Differentiability:**
L(w) is smooth and differentiable everywhere. The gradient ∇_w L = -(2/N)Xᵀ(y - Xw) exists at every point in parameter space without exception.

**Property 3 — Linear gradient equation:**
When we set ∇_w L = 0:
-(2/N)Xᵀ(y - Xw) = 0
→ XᵀXw = Xᵀy

This is a **linear system** in w. The unknowns (w) appear only as first-degree terms — no w², no sin(w), no w·w products. A linear system can be solved directly by matrix inversion: w* = (XᵀX)⁻¹Xᵀy.

The combination of all three means: we know a minimum exists (convex), we can find where gradient = 0 (differentiable), and the equation gradient = 0 can be solved algebraically in one step (linear in w).

**Part 3:**
**Four models without closed-form solutions:**

1. **Logistic Regression:**
Gradient: ∇_w L = Σₙ (σ(wᵀxₙ) - yₙ)xₙ = 0
Why fails: σ(wᵀxₙ) = 1/(1+exp(-wᵀxₙ)) is a nonlinear function of w. Setting the gradient to zero gives Σₙ σ(wᵀxₙ)xₙ = Σₙ yₙxₙ — the sigmoid function wraps around w in a way that cannot be algebraically inverted to isolate w.

2. **Neural Networks:**
Gradient involves chains of nonlinear activation functions (sigmoid, ReLU, tanh) applied in multiple layers. The composition of nonlinear functions creates a highly nonlinear, non-convex loss landscape. Setting gradients to zero gives an enormously complex nonlinear system with no analytical solution. Additionally, the loss is non-convex — multiple local minima exist.

3. **Lasso Regression:**
L(w) = (1/2N)||y - Xw||² + λΣⱼ|wⱼ|
The L1 penalty |wⱼ| is not differentiable at wⱼ = 0. Setting the subgradient to zero requires checking whether each weight is at its kink point or not — this cannot be solved as a simple linear system. Requires specialized algorithms (coordinate descent, LARS).

4. **SVM with kernel:**
The SVM optimization involves Lagrange multipliers with inequality constraints. Setting KKT conditions to zero yields a quadratic programming problem — not a simple linear equation in parameters. Requires specialized convex QP solvers.

---

# SECTION 2: Gradient Descent Foundations

---

**Question 2**
Gradient Descent is the foundational optimization algorithm for ML.

1. Explain precisely what gradient descent is. What is the gradient telling us, and why do we subtract it (rather than add it) to reduce the loss?
2. Write the gradient descent update rule. Explain every component and derive it intuitively from the concept of "moving in the direction of steepest descent."
3. Explain the two mathematical requirements a function must satisfy for gradient descent to be applicable. Give an example of a function that violates each requirement and explain what goes wrong.

### ✅ Answer

**Part 1:**
**Precise explanation of gradient descent:**

Gradient descent is an **iterative first-order optimization algorithm** that minimizes a function by repeatedly moving in the direction of steepest descent.

**What the gradient tells us:**
The gradient ∇L(θ) is a vector pointing in the direction of **steepest ascent** of the loss function at the current parameter location θ. Each component ∂L/∂θⱼ tells you: "if I increase θⱼ by a tiny amount, the loss increases by this rate."

**Why we SUBTRACT the gradient:**
We want to MINIMIZE the loss (descent, not ascent). If ∇L points uphill, then -∇L points downhill — toward lower loss values. Moving in the -∇L direction decreases the loss most rapidly at the current point.

**Formal derivation:**
By first-order Taylor expansion:
L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀ Δθ

To decrease L, we want ∇L(θ)ᵀ Δθ < 0 (the inner product negative). This is maximally negative when Δθ = -η∇L(θ) — pointing exactly opposite to ∇L.

**Part 2:**
**Update rule and intuition:**

**θ^(t+1) = θ^(t) - η ∇L(θ^(t))**

Every component:
- **θ^(t):** Current parameter vector at iteration t — where we are on the loss landscape right now
- **∇L(θ^(t)):** Gradient of the loss evaluated at current parameters — the direction pointing uphill (toward higher loss)
- **-∇L(θ^(t)):** Negative gradient — the direction pointing downhill (toward lower loss)
- **η (eta):** Learning rate — how large a step to take in the downhill direction
- **θ^(t+1):** New parameter vector — one step downhill from current position

**Intuitive derivation:**
Imagine standing on a hilly landscape (the loss surface). You want to reach the valley (minimum). At each step:
1. Look around and identify which direction is downhill (compute -∇L)
2. Take a step of size η in that direction
3. Find yourself at a new, lower position
4. Repeat

The gradient is a local property — it tells you downhill at THIS point, not globally. So you take a small step (controlled by η), recompute the gradient at the new location, and take another step. Over many iterations, you descend toward the minimum.

**Part 3:**
**Two requirements and violations:**

**Requirement 1 — Differentiability:**
The gradient ∇L must exist at every point encountered during optimization. Gradient descent literally computes and uses ∇L — if this doesn't exist, the algorithm breaks.

**Violation example — MAE (Mean Absolute Error):**
L(w) = (1/N)Σₙ |yₙ - wᵀxₙ|

The absolute value |·| is not differentiable at 0. When a residual yₙ - wᵀxₙ = 0, the derivative doesn't exist (left derivative = -1, right derivative = +1, no unique gradient). Standard gradient descent cannot proceed at this point — it doesn't know which direction to step.

**What goes wrong:** The algorithm may oscillate around the exact minimum, never landing on it, because the gradient "flips sign" on either side of the non-differentiable point.

**Requirement 2 — Convexity:**
Convexity guarantees that any local minimum is the global minimum. Without convexity, gradient descent may converge to a local minimum that is much worse than the global minimum.

**Violation example — Neural Network loss:**
Neural network losses are highly non-convex — they have many local minima, saddle points, and flat regions. The loss surface might look like a mountain range with many valleys at different depths.

**What goes wrong:** Gradient descent finds a local minimum (the nearest valley) which may be arbitrarily worse than the global minimum. The algorithm has no way to know whether it has found the best valley or just a nearby one. Different random initializations lead to different local minima with different generalization properties.

---

**Question 3**
The learning rate is the most critical hyperparameter in gradient descent.

1. Explain what the learning rate η does. Describe all four scenarios: η too large (diverges), η too large (bounces but converges slowly), η too small (converges slowly), and η just right.
2. Explain the dilemma: why can't you always use a very small learning rate to be safe? What is the practical cost of being too conservative?
3. Explain Learning Rate Scheduling as a solution to this dilemma. Describe Step Decay, Exponential Decay, and Cosine Annealing — for each, state its formula concept, key advantage, and key disadvantage.

### ✅ Answer

**Part 1:**
**Four scenarios for learning rate:**

The update rule: θ^(t+1) = θ^(t) - η ∇L(θ^(t))

The step size at each iteration = η × ||∇L||

**Scenario 1 — η too large (diverges):**
Each step overshoots the minimum by so much that the new loss is HIGHER than the old loss. The next gradient then points back even more strongly. The parameters "fly off to infinity" — loss increases without bound. Like trying to descend a hill by taking giant leaps that send you flying off the other side.

Visual: Loss curve shows monotonically increasing values — catastrophic failure.

**Scenario 2 — η slightly too large (oscillates/bounces slowly):**
Steps overshoot the minimum but not catastrophically — they land on the other side. The algorithm oscillates back and forth across the minimum, making very slow overall progress toward it. The loss decreases on average but very slowly with high variance between steps.

Visual: Loss curve shows jagged, oscillating pattern that slowly decreases.

**Scenario 3 — η too small (converges slowly):**
Each step is tiny — the algorithm makes minuscule progress per iteration. Technically converges to the correct minimum, but requires an enormous number of iterations. Each iteration costs the same computational resources regardless of step size. Total training time = (iterations needed) × (cost per iteration) → very high total cost.

Additional risk: In non-convex settings, tiny steps may get trapped on flat plateaus or shallow local minima because the noise is too small to escape.

Visual: Loss curve shows smooth but extremely gradual decrease over many epochs.

**Scenario 4 — η just right:**
Each step makes meaningful progress toward the minimum without overshooting. Loss decreases smoothly and consistently. Converges in a reasonable number of iterations. The ideal balance between progress per step and stability.

**Part 2:**
**Why you can't always use very small η:**

**Computational cost:**
With N=1,000,000 training examples and D=1,000 features, each gradient computation requires O(N×D) = 10⁹ operations. If η is 100× too small, you need 100× more iterations to achieve the same convergence. Total additional cost: 100× the compute time.

**Flat regions and saddle points:**
In non-convex landscapes (neural networks), the gradient is often very small in flat regions or near saddle points. With a tiny η, the algorithm makes negligible progress per step — it effectively "gets stuck" even though a larger step would escape the flat region and continue descending.

**Implicit regularization:**
The path taken during optimization matters for generalization (see flat vs. sharp minima). Very conservative updates take a very direct path to the nearest minimum — which may be a sharp, poorly generalizing one. More dynamic updates (larger η) explore more of the landscape and tend to find flatter, better-generalizing minima.

**Practical cost:**
Deep learning models (GPT, ResNet) train for days or weeks. Using η = 0.0001 instead of optimal η = 0.01 would extend training from 1 week to 100 weeks — completely impractical.

**Part 3:**
**Learning Rate Scheduling:**

**Core strategy:** Start large (explore landscape quickly, escape bad initial regions) → progressively decay (fine-tune weights, converge precisely).

**1. Step Decay:**
Concept: Multiply η by a factor (e.g., 0.1) every N epochs.
Formula: η(epoch) = η₀ × γ^(floor(epoch/step_size))

Advantage: Highly intuitive — creates distinct "phases" of training. Initial rapid learning phase, then sudden shift to fine-tuning phase. Easy to understand and explain.

Disadvantage: The abrupt drop in learning rate causes a sudden "shock" to training dynamics — the optimizer suddenly takes much smaller steps, which can temporarily destabilize training. Requires manual specification of two hyperparameters: exactly when to drop (step_size) and by how much (gamma).

**2. Exponential Decay:**
Concept: Multiply η by a constant factor at every epoch.
Formula: η(t) = η₀ × γᵗ (where γ < 1, e.g., γ = 0.99)

Advantage: Smooth and continuous — avoids jarring discontinuities of step decay. Learning rate decreases gradually at every epoch, naturally narrowing the search region over time.

Disadvantage: If γ is even slightly too small (too aggressive), learning rate shrinks to near-zero very early in training before the model has found a good minimum. Sensitive to the choice of γ.

**3. Cosine Annealing:**
Concept: Modulate η following a cosine curve from η_max to η_min over T_max epochs.
Formula: η(t) = η_min + (1/2)(η_max - η_min)(1 + cos(πt/T_max))

Advantage: Excellent empirical performance. Profile:
- Drops SLOWLY at first (allows escaping bad initial regions)
- Drops RAPIDLY in the middle (traverses the landscape efficiently)
- Slows to a crawl at the end (precise final convergence)
Can be combined with "warm restarts" to escape local minima periodically.

Disadvantage: Requires knowing T_max (total training epochs) upfront to scale the cosine curve. Less flexible if using early stopping or training for variable duration.

---

# SECTION 3: Types of Gradient Descent

---

**Question 4**
Three variants of gradient descent make different tradeoffs.

1. Explain Batch Gradient Descent (BGD). Write its update rule, explain what data it uses per update, and describe its convergence behavior. When is it practical?
2. Explain Stochastic Gradient Descent (SGD). Write its update rule, explain what data it uses, and explain why its noisy updates are sometimes BENEFICIAL rather than just a limitation.
3. Explain Mini-Batch Gradient Descent (MBGD). Write its update rule and explain why it is the industry standard by describing the specific tradeoffs it makes between BGD and SGD.

### ✅ Answer

**Part 1:**
**Batch Gradient Descent (BGD):**

**Update rule:**
θ^(t+1) = θ^(t) - η × (1/N) Σₙ₌₁ᴺ ∇Lₙ(θ^(t))

Uses ALL N training examples to compute a single gradient update.

**Data per update:** The entire training dataset — N examples per update.

**Convergence behavior:**
- Uses the exact gradient (not an estimate) — the most accurate direction at each step
- Smooth, monotonically decreasing loss curve (for convex functions)
- Stable updates — the direction doesn't change randomly between iterations
- For quadratic loss (linear regression): converges in O(1/t) iterations

**Practical usage:**
- Linear regression and logistic regression with small-to-medium datasets
- Problems where exact gradient is critical
- **NOT practical for large N** (N > 100,000): must process all N examples per step. For N = 10 million, each update requires 10 million gradient computations — minutes per update step.

**Part 2:**
**Stochastic Gradient Descent (SGD):**

**Update rule:**
θ^(t+1) = θ^(t) - η × ∇Lₙ(θ^(t)) (for one randomly chosen n)

Uses ONE randomly selected training example per update.

**Convergence behavior:**
- Uses a noisy estimate of the gradient — very cheap per update but highly variable direction
- Loss curve is noisy and jagged — individual steps may increase loss
- Converges on average toward minimum but oscillates around it
- Requires decaying learning rate to converge properly (otherwise keeps oscillating)

**Why noise is BENEFICIAL:**

1. **Escape local minima:** In non-convex landscapes (neural networks), the true gradient at a local minimum is exactly zero — batch GD would get permanently stuck. SGD's noise means the estimated gradient from one sample is non-zero even at a local minimum, allowing the algorithm to escape.

2. **Escape saddle points:** Similar argument — saddle points have zero gradient but SGD's noise provides random perturbations that kick it off the saddle.

3. **Flat minima preference:** The noise in SGD acts as an implicit regularizer — the algorithm tends to bounce out of narrow, sharp minima (which have steep walls that cause large gradient oscillations) and settle in wide, flat minima (which are more forgiving of perturbations). Flat minima generalize better.

4. **Online learning:** Can process streaming data one example at a time — model updates continuously with new data without storing the whole dataset.

**Part 3:**
**Mini-Batch Gradient Descent (MBGD):**

**Update rule:**
θ^(t+1) = θ^(t) - η × (1/B) Σₙ∈batch ∇Lₙ(θ^(t))

Uses B randomly selected examples per update (typically B = 32, 64, 128, 256).

**Why it is the industry standard:**

| Property | BGD | SGD | MBGD (B=64) |
|----------|-----|-----|-------------|
| Gradient accuracy | Exact | Very noisy | Moderate noise |
| Cost per update | O(N×D) | O(D) | O(B×D) |
| Iterations to converge | Few | Many | Moderate |
| GPU utilization | Full (but slow) | Poor (sequential) | Excellent (parallelism) |
| Memory requirement | All N examples | 1 example | B examples |
| Noise for escaping | None | Too much | Just right |

**The specific tradeoffs:**

1. **Speed vs. stability:** B is the tradeoff knob between SGD (fastest per update, least stable) and BGD (slowest per update, most stable). B=32-256 achieves both reasonable stability and fast updates.

2. **GPU parallelism:** Modern GPUs are optimized for matrix operations. Computing the gradient for B examples simultaneously using matrix operations is nearly as fast as computing for 1 example — but gives B× more information. This is the core reason MBGD dominates in deep learning.

3. **Moderate noise:** Enough noise to escape local minima and saddle points (unlike BGD) but stable enough to converge reliably (unlike SGD with very small B).

4. **Memory efficiency:** Only B examples in memory at once vs. all N for BGD — critical for large datasets.

---

# SECTION 4: Second-Order Methods

---

**Question 5**
Newton's Method and IRLS use curvature information for faster optimization.

1. Explain the core idea of second-order optimization. What does the Hessian ∇²L add beyond what the gradient provides, and how does Newton's method use it?
2. Write Newton's update rule for logistic regression. Explain what each component represents and why this method converges faster than gradient descent.
3. Explain the IRLS (Iteratively Reweighted Least Squares) algorithm. What does each step do, and why is it called "Iteratively Reweighted"?

### ✅ Answer

**Part 1:**
**Second-order optimization concept:**

First-order methods (gradient descent) use only ∇L — the direction of steepest descent. But they ignore curvature — how steeply the function curves in different directions.

**What the Hessian adds:**
∇²L(θ) is the matrix of second derivatives — it tells you not just which way is downhill, but **how the slope is changing**:
- In directions with high curvature (steep bowl): take small steps — the loss changes rapidly, and overshooting is dangerous
- In directions with low curvature (flat plateau): take large steps — the loss changes slowly, and you can move far without risk

**The analogy:**
- Gradient: "This way is downhill"
- Hessian: "...and it's a gentle slope" (take a big step) or "...and it's a cliff" (take a tiny step)

**Newton's step interpretation:**
Newton's method approximates the loss with a quadratic (second-order Taylor expansion) and jumps directly to the minimum of that quadratic approximation:

θ^(t+1) = θ^(t) - [∇²L(θ^(t))]⁻¹ ∇L(θ^(t))

The Hessian inverse automatically scales the step: large step in low-curvature directions, small step in high-curvature directions. No learning rate needed!

**Part 2:**
**Newton's update for logistic regression:**

**Hessian for logistic regression:**
∇²L(w) = (1/N) Σₙ pₙ(1-pₙ) xₙxₙᵀ = (1/N) XᵀRX

Where R = diag(p₁(1-p₁), ..., pₙ(1-pₙ)) is a diagonal matrix of weights.

**Newton update rule:**
w^(t+1) = w^(t) - [XᵀRX]⁻¹ Xᵀ(p - y)

Every component:
- **Xᵀ(p - y):** The gradient — weighted sum of prediction errors
- **XᵀRX:** The Hessian — weighted covariance of features, where each point is weighted by pₙ(1-pₙ) (its uncertainty)
- **[XᵀRX]⁻¹:** Hessian inverse — scales the gradient by curvature information
- **pₙ(1-pₙ):** Maximum at pₙ=0.5 (maximum uncertainty) — uncertain predictions have highest curvature and smallest Newton steps

**Why converges faster than gradient descent:**

1. **Automatic step size:** The Hessian inverse automatically determines the optimal step size in each direction — no learning rate hyperparameter needed.

2. **Quadratic convergence:** Near the minimum, Newton's method converges quadratically — the number of correct digits roughly doubles at each iteration. Gradient descent converges linearly (error reduces by a constant factor each step).

3. **Accounts for correlations:** The Hessian captures how changing one parameter affects another. In ill-conditioned problems (highly correlated features), gradient descent zigzags inefficiently, while Newton's method follows a direct path.

**Cost:** Each Newton step requires computing and inverting the D×D Hessian: O(D²) for computation, O(D³) for inversion. For D=10,000 features: 10⁸ storage and 10¹² operations per step — prohibitively expensive for large problems.

**Part 3:**
**IRLS Algorithm:**

IRLS reformulates Newton's update as a sequence of weighted least squares problems.

**Rewriting the Newton update:**
w^(new) = (XᵀRX)⁻¹ Xᵀ R z

Where z is the "adjusted response" (pseudo-targets):
zₙ = wᵀxₙ + R_n⁻¹(yₙ - pₙ) = wᵀxₙ + (yₙ - pₙ)/(pₙ(1-pₙ))

**Why this is a least squares problem:**
(XᵀRX)⁻¹Xᵀ R z is exactly the solution to the **weighted least squares problem**:
min_w (z - Xw)ᵀ R (z - Xw)

This minimizes weighted squared error between pseudo-targets z and predictions Xw, where each point n is weighted by Rₙ = pₙ(1-pₙ).

**The IRLS algorithm step by step:**
1. **Initialize w** (e.g., w = 0)
2. **Compute predictions:** pₙ = σ(wᵀxₙ) for all n
3. **Compute weights:** Rₙ = pₙ(1-pₙ) for all n — higher weight for uncertain predictions
4. **Construct pseudo-targets:** zₙ = wᵀxₙ + (yₙ-pₙ)/Rₙ — adjusted response for each point
5. **Solve weighted least squares:** w ← (XᵀRX)⁻¹XᵀRz — one linear regression solve
6. **Repeat 2-5 until convergence**

**Why "Iteratively Reweighted":**
At each iteration, the weights Rₙ = pₙ(1-pₙ) change because predictions pₙ change with each update to w. We are repeatedly solving least squares problems, but with different weights each time — hence "Iteratively Reweighted" Least Squares.

---

**Question 6**
Compare IRLS and Gradient Descent comprehensively.

1. Create a complete comparison table of IRLS vs. Gradient Descent across: optimization order, step size, convergence speed, computational cost per iteration, scalability, and practical application domain.
2. Explain why IRLS is preferred for small/medium classical statistics problems while gradient descent dominates large-scale ML. What is the specific computational bottleneck for each?
3. For a logistic regression problem with D=10 features and N=500 examples, which would you use and why? Repeat for D=10,000 features and N=10,000,000 examples.

### ✅ Answer

**Part 1:**
**Complete comparison table:**

| Aspect | Gradient Descent | IRLS |
|--------|-----------------|------|
| **Optimization order** | First-order (gradient only) | Second-order (gradient + Hessian) |
| **Step size** | Manual — requires η hyperparameter | Automatic — determined by Hessian inverse |
| **Convergence speed** | Linear convergence | Quadratic convergence (near optimum) |
| **Cost per iteration** | O(N×D) | O(N×D² + D³) |
| **Memory** | O(D) for gradient | O(D²) for Hessian |
| **Scalability** | Excellent (N and D scale fine) | Poor (D³ cubic in dimensions) |
| **Learning rate tuning** | Required (major challenge) | None needed |
| **Theoretical guarantees** | Convergence for convex with right η | Strong — Newton's method theory |
| **Non-convex problems** | Works (with caveats) | Dangerous — may diverge |
| **Numerical stability** | Generally stable | Can be unstable if Hessian nearly singular |
| **Application domain** | Large-scale ML, deep learning | Classical stats, small/medium problems |

**Part 2:**
**Why each dominates its domain:**

**IRLS preferred for classical statistics:**
- D is typically small (tens to hundreds of features, not millions)
- N is moderate (thousands to tens of thousands)
- Hessian computation O(N×D²) and inversion O(D³) are feasible
- Quadratic convergence means very few iterations (often 5-20 iterations to convergence vs. thousands for GD)
- No hyperparameter tuning (no η to select) — important in classical statistics where reproducibility and not requiring extensive tuning is valued
- Strong theoretical guarantees align with statistical inference goals

**Computational bottleneck for IRLS:**
D³ matrix inversion. For D=1000: 10⁹ operations. For D=10,000: 10¹² operations per step. For D=100,000: 10¹⁵ operations — years of compute. IRLS is completely infeasible when D is large.

**Gradient descent dominates large-scale ML:**
- D can be millions (neural networks with millions of parameters)
- N can be billions (internet-scale datasets)
- Each gradient step costs O(N×D) — or O(B×D) per mini-batch step
- No D² or D³ terms — scales linearly in D for each update
- The η hyperparameter is manageable with learning rate scheduling and adaptive optimizers

**Computational bottleneck for gradient descent:**
N×D per update (full batch) or B×D per mini-batch update. Mini-batch reduces this to a constant O(B×D) per step, independent of N — enabling massive scale. The bottleneck is total iterations × cost per step, not a single catastrophic D³ term.

**Part 3:**
**Case 1: D=10, N=500 (small classical problem):**

**Use IRLS.**

Reasoning:
- Hessian is 10×10 — trivial to compute and invert (1000 operations)
- Total IRLS cost per iteration: O(500×100 + 1000) = O(51,000) — negligible
- IRLS converges in ~5-20 iterations → total cost: ~10⁶ operations
- Gradient descent would need 1000+ iterations → similar or higher total cost
- IRLS has no η to tune — simpler, more reliable
- Strong theoretical guarantees relevant for statistical inference

**Case 2: D=10,000, N=10,000,000 (massive ML problem):**

**Use Mini-Batch Gradient Descent (with Adam optimizer).**

Reasoning:
- IRLS Hessian: 10,000×10,000 = 10⁸ entries — requires 800 MB storage just for the matrix
- IRLS Hessian inversion: O(D³) = O(10¹²) operations per step — thousands of CPU hours per iteration
- Gradient descent (mini-batch, B=256): O(B×D) = O(2,560,000) per step — milliseconds
- Learning rate can be managed with Adam (adaptive, minimal tuning)
- With proper η schedule, gradient descent converges in thousands of iterations but total cost is feasible

IRLS is simply computationally infeasible at this scale — the D³ bottleneck is an absolute barrier regardless of how many iterations are needed.

---

# SECTION 5: Adaptive Optimizers

---

**Question 7**
Standard SGD has fundamental limitations that motivated adaptive optimizers.

1. Explain the four specific limitations of standard SGD. For each, give a concrete example of a problem where that limitation causes poor performance.
2. Explain the core insight of adaptive optimizers: "Not all parameters should move at the same speed." Why is a single global learning rate suboptimal?
3. Explain Momentum. Write the update rule, explain the velocity term, and use the "heavy ball" physics analogy to describe how it overcomes ravines and local optima.

### ✅ Answer

**Part 1:**
**Four limitations of standard SGD:**

1. **Single global learning rate η:**
All D parameters share the same step size, even though different parameters may need very different amounts of adjustment. 
Concrete example: A text classification model with 50,000 word features. Common words ("the", "a") appear in almost every training example — their corresponding weights get gradient updates at every step. Rare technical terms appear in 0.01% of examples — their weights receive updates very rarely. A single η that's good for common words is too large for stable sparse updates; an η good for rare words means common words learn agonizingly slowly.

2. **Same step size for all parameters:**
Parameters whose loss is very sensitive to their value (high curvature) need small steps; parameters with low sensitivity (low curvature) can take large steps.
Concrete example: In a neural network, the weights in the first layer receive tiny gradients (vanishing gradient problem) while weights in the last layer receive large gradients. A step size appropriate for the last layer causes the first layer to move in infinitesimally tiny amounts — learning effectively stalls in early layers.

3. **Sensitive to feature scaling:**
If feature x₁ has values in [0,1] and feature x₂ has values in [0,1,000,000], the gradient with respect to w₂ is 10⁶× larger than the gradient with respect to w₁. The same η will cause enormous updates to w₂ (potentially diverging) and tiny updates to w₁ (extremely slow learning).
Concrete example: Predicting house prices with features like square footage (500-5000) and number of bathrooms (1-5). Without scaling, the gradient is dominated by square footage, causing numerical instability.

4. **Struggles with ill-conditioned problems:**
In ill-conditioned settings (Hessian has very different eigenvalues in different directions — some directions have high curvature, others very low), a single η is either too large for the high-curvature directions (oscillates/diverges) or too small for the low-curvature directions (extremely slow).
Concrete example: Training a neural network on images where early layers learn low-level features (many correlated features, high ill-conditioning). Standard SGD zigzags in the high-curvature directions while barely moving in the low-curvature ones.

**Part 2:**
**Why single global η is suboptimal:**

The optimal step size for parameter θⱼ depends on the local curvature ∂²L/∂θⱼ²:
- High curvature (loss changes rapidly with θⱼ): optimal step size ∝ 1/(curvature) — small step
- Low curvature (loss barely changes with θⱼ): optimal step size ∝ 1/(curvature) — large step

A global η must be set conservatively enough for the highest-curvature parameter — this makes it far too small for all low-curvature parameters. The algorithm spends most of its time making tiny, inefficient steps in directions where it could safely move fast.

Adaptive optimizers estimate the curvature (or a proxy for it) for each parameter individually using gradient history, then set per-parameter learning rates: effectively η_j = η/√(estimated_curvature_j). This allows each parameter to move at its own optimal speed.

**Part 3:**
**Momentum:**

**Update rule:**
v^(t+1) = γ v^(t) + η ∇L(θ^(t))
θ^(t+1) = θ^(t) - v^(t+1)

Where γ ≈ 0.9 is the momentum coefficient.

**The velocity term v^(t):**
v^(t) is a weighted sum of all past gradients (with exponential decay):
v^(t) = η∇L^(t) + γη∇L^(t-1) + γ²η∇L^(t-2) + ...

Recent gradients contribute more (weight = 1), older gradients less (weight = γᵏ → 0 exponentially). This is an **exponentially weighted moving average** of past gradients.

**The heavy ball physics analogy:**
Imagine rolling a heavy bowling ball down a hilly loss landscape:

**Without momentum (standard SGD):**
Like rolling a ping-pong ball — it stops dead at every small bump and changes direction instantly based only on the current slope. In a "ravine" (narrow valley with steep walls and gentle floor), the ball oscillates wildly across the ravine's width (responding to steep cross-ravine gradient) while making tiny progress along the ravine's length (gentle gradient in the right direction).

**With momentum (heavy ball):**
- **Accumulates velocity:** The ball gains speed as it consistently rolls in the same direction. After many steps in the same direction, v grows large → moves faster and faster in that direction.
- **Opposing gradients cancel:** In the ravine, the cross-ravine oscillations cancel out — the gradient alternates left-right each step, and v_left + v_right ≈ 0. The ball stops oscillating across the ravine.
- **Aligned gradients accumulate:** The along-ravine gradient consistently points the same direction — v accumulates → the ball accelerates along the ravine floor.
- **Powers through local optima:** A shallow local minimum has steep walls but the heavy ball has enough velocity to roll through it and continue to a deeper minimum — the momentum carries it past small bumps.

---

**Question 8**
AdaGrad, RMSProp, and Adam represent the evolution of adaptive optimizers.

1. Explain AdaGrad. Write its update rule, explain what it accumulates, and explain the specific failure mode that limits it.
2. Explain RMSProp as AdaGrad's fix. Write its update rule, explain the exponential moving average, and explain what problem this solves.
3. Explain Adam as Momentum + RMSProp. Write its complete update rule including bias correction. Explain why bias correction is necessary and what β₁ and β₂ control.

### ✅ Answer

**Part 1:**
**AdaGrad (Adaptive Gradient Algorithm):**

**Update rule:**
G^(t)_j = G^(t-1)_j + (∂L/∂θ_j)² — accumulate squared gradients
θ^(t+1)_j = θ^(t)_j - (η/√(G^(t)_j + ε)) × ∂L/∂θ_j

Where ε is a small constant (e.g., 10⁻⁸) for numerical stability.

**What it accumulates:**
G_j = Σₜ (∂L/∂θ_j)^(t)² — the sum of all squared gradients for parameter j from the beginning of training.

**The effective learning rate for parameter j:**
η_j^(eff) = η/√(G_j) — inversely proportional to the square root of accumulated gradient magnitude

- Parameters with consistently large gradients: G_j grows large → η_j^(eff) decreases → smaller steps (prevents overshooting)
- Parameters with consistently small gradients (sparse features): G_j stays small → η_j^(eff) stays large → larger steps (accelerates learning for rare features)

**The failure mode:**
G_j is a sum that **grows forever** — it only accumulates, never decreases or forgets old information. After many training steps:
- G_j → ∞ for all parameters
- η_j^(eff) = η/√G_j → 0
- The effective learning rate approaches zero
- The optimizer **freezes** — makes no more progress

After enough training steps, AdaGrad effectively stops learning regardless of how much signal remains in the gradients. Good for convex problems (which converge quickly before freezing) and sparse data, but fails for deep learning with long training runs.

**Part 2:**
**RMSProp (Root Mean Square Propagation):**

**Update rule:**
v^(t)_j = β v^(t-1)_j + (1-β)(∂L/∂θ_j)² — exponential moving average
θ^(t+1)_j = θ^(t)_j - (η/√(v^(t)_j + ε)) × ∂L/∂θ_j

Where β ≈ 0.9 is the decay factor.

**The exponential moving average:**
Instead of summing ALL past squared gradients, RMSProp uses a weighted average that **forgets old information**:
v^(t) = β × v^(t-1) + (1-β) × g²_t

The weight assigned to gradient from k steps ago decays as β^k:
v^(t) = (1-β)[g²_t + β g²_{t-1} + β² g²_{t-2} + ...]

For β=0.9, gradients from 10 steps ago have weight 0.9^10 ≈ 0.35 — still relevant but decreasing. Gradients from 100 steps ago have weight 0.9^100 ≈ 0.000027 — effectively forgotten.

**What problem this solves:**
RMSProp fixes AdaGrad's freezing problem. Because old gradients are forgotten (exponential decay), v_j doesn't grow without bound — it stabilizes at a running estimate of recent gradient magnitudes. The effective learning rate η/√v_j remains non-zero throughout training, enabling long training runs.

**Effect:** The effective learning rate for each parameter is inversely proportional to the magnitude of RECENT (not all-time) gradients, automatically rescaling curvature and stabilizing training for ill-conditioned problems.

**Part 3:**
**Adam (Adaptive Moment Estimation):**

Adam combines Momentum (tracks first moment — mean of gradients) and RMSProp (tracks second moment — variance of gradients):

**Track two moments:**
m^(t) = β₁ m^(t-1) + (1-β₁) g^(t) — first moment (gradient mean, "Momentum")
v^(t) = β₂ v^(t-1) + (1-β₂) g^(t)² — second moment (gradient variance, "RMSProp")

**Bias-corrected estimates:**
m̂^(t) = m^(t) / (1 - β₁^t)
v̂^(t) = v^(t) / (1 - β₂^t)

**Update rule:**
θ^(t+1) = θ^(t) - η × m̂^(t) / (√v̂^(t) + ε)

**Why bias correction is necessary:**
At initialization, m^(0) = 0 and v^(0) = 0. In the early iterations, the exponential moving averages are biased toward zero because they haven't "warmed up" yet.

Example: At t=1 with β₁=0.9, first gradient g^(1):
m^(1) = 0.9×0 + 0.1×g^(1) = 0.1×g^(1)

The EMA value 0.1×g^(1) severely underestimates the true gradient g^(1) by a factor of 10. Dividing by (1-β₁^1) = (1-0.9) = 0.1 gives: m̂^(1) = 0.1g^(1)/0.1 = g^(1). Bias correction restores the proper scale.

As t → ∞: β₁^t → 0 → (1-β₁^t) → 1 → bias correction becomes negligible. The correction only matters in the early iterations.

**What β₁ and β₂ control:**
- **β₁ ≈ 0.9:** Controls momentum decay — how much of the velocity is retained from previous steps. Higher β₁ = longer memory = smoother updates but slower response to changing gradient direction.
- **β₂ ≈ 0.999:** Controls RMSProp decay — how quickly old second-moment estimates are forgotten. Higher β₂ = longer memory for gradient variance = more stable but slower adaptation to changing curvature.

Default values (β₁=0.9, β₂=0.999) work well across a wide range of problems — a key reason for Adam's popularity.

---

**Question 9**
Nesterov Accelerated Gradient (NAG) improves upon basic Momentum.

1. Explain the specific problem with basic Momentum that NAG addresses. Use the "heavy ball overshooting" analogy.
2. Write the NAG update rule and explain how the "lookahead" gradient computation differs from standard Momentum. What does computing the gradient at θ - γv rather than θ achieve?
3. Compare standard Momentum and NAG in terms of convergence behavior near a minimum. Which is more likely to overshoot and why?

### ✅ Answer

**Part 1:**
**The problem with basic Momentum:**

Standard Momentum: v^(t+1) = γv^(t) + η∇L(θ^(t)), then step

The heavy ball analogy shows the problem clearly:

Imagine a ball rolling down a valley. With Momentum, the ball accumulates speed as it descends. When approaching the minimum at the valley floor:
1. The ball is moving fast (high velocity from accumulated momentum)
2. The gradient at the current position says "slow down — you're approaching the bottom"
3. But the gradient is computed at the CURRENT position, and the step is already determined by the accumulated velocity
4. The ball continues past the minimum, overshooting to the other side
5. It oscillates back and forth around the minimum, each time overshooting

The problem: Momentum computes the gradient at the current position BEFORE moving. By the time the gradient is used, the ball has already moved to a different location due to its velocity. The gradient correction comes too late.

**Real consequence:** Standard Momentum can oscillate around the minimum for many iterations before converging, especially in narrow valleys or when the momentum coefficient γ is large.

**Part 2:**
**NAG update rule and lookahead:**

**Standard Momentum:**
v^(t+1) = γv^(t) + η∇L(**θ^(t)**)  ← gradient at CURRENT position
θ^(t+1) = θ^(t) - v^(t+1)

**NAG (Nesterov Accelerated Gradient):**
v^(t+1) = γv^(t) + η∇L(**θ^(t) - γv^(t)**)  ← gradient at ANTICIPATED position
θ^(t+1) = θ^(t) - v^(t+1)

**What θ^(t) - γv^(t) represents:**
This is where the momentum term ALONE would take you from the current position — the "anticipated future position" based purely on current velocity. NAG computes the gradient at this lookahead point before actually making the step.

**What this achieves:**
If you're about to overshoot the minimum (the lookahead position is past the minimum), the gradient at the lookahead position will point BACKWARD — opposing the direction of motion. This gradient signal acts as a brake, reducing the velocity before the full step is taken.

NAG is like a skier who looks ahead to where they're going (not just where they are) and starts braking early if they see a cliff approaching, rather than braking only after they've arrived at the edge.

**Part 3:**
**Momentum vs. NAG near a minimum:**

**Standard Momentum near minimum:**
- At position θ close to θ*, gradient ∇L(θ) is small but pointing back toward θ*
- Velocity v^(t) is large (accumulated from previous steps)
- Update: v^(t+1) = γv^(t) + η∇L(θ) ≈ γv^(t) (gradient is small, velocity dominates)
- The step θ → θ - v^(t+1) is large, overshooting the minimum
- The model continues past θ* to the other side

**NAG near minimum:**
- Lookahead position: θ - γv is already past θ* (or close to it)
- Gradient at lookahead: ∇L(θ - γv) points back toward θ* more strongly than ∇L(θ)
- This correction directly reduces v^(t+1): the backward gradient from the lookahead position counteracts the forward momentum
- The actual step is smaller — the ball slows down before reaching the minimum
- Less overshooting, more precise convergence

**Which overshoots more:**
Standard Momentum overshoots more because it uses a "stale" gradient from the current position before momentum takes effect. NAG uses a "predictive" gradient that accounts for where momentum will carry the model — allowing proactive braking.

Theoretical result: NAG achieves convergence rate O(1/t²) for convex functions vs. O(1/t) for gradient descent without momentum — a provably better theoretical rate, largely due to this lookahead correction mechanism.

---

# SECTION 6: Non-Convex Optimization

---

**Question 10**
Non-convex optimization in deep learning presents unique challenges and insights.

1. Explain why local minima are less of a problem in high-dimensional deep learning than the 2D illustrations suggest. What is the geometric argument about the probability of encountering a true local minimum?
2. Explain what saddle points are and why they are the real enemy in high-dimensional optimization. How do mini-batch SGD and Momentum help escape them?
3. Explain the distinction between "sharp minima" and "flat minima." Why do flat minima generalize better, and why does SGD+Momentum tend to find flat minima while Adam might find sharp minima?

### ✅ Answer

**Part 1:**
**Why local minima are less problematic in high dimensions:**

**The 2D intuition (misleading):**
In 2D, a local minimum requires the function to curve upward in BOTH directions from the critical point. Visually, these are common — every valley in a 2D landscape is a local minimum.

**The high-dimensional reality:**
For a critical point (∇L = 0) to be a true local minimum, the Hessian ∇²L must be positive definite — the curvature must be POSITIVE in ALL dimensions simultaneously.

**The geometric argument:**
In a D-dimensional space, each dimension's curvature is independently either positive or negative at a random critical point. If we assume each dimension's curvature is independently ±1 with equal probability, the probability that ALL D dimensions have positive curvature = (1/2)^D.

For D = 1,000,000 parameters: P(local minimum) = (1/2)^(1,000,000) ≈ 10^(-300,000) — essentially zero.

**What actually exists:**
Most critical points in high-dimensional non-convex landscapes are **saddle points** — where the curvature is positive in some dimensions and negative in others. A saddle point with 500,000 positive and 500,000 negative curvature dimensions is astronomically more likely than all 1,000,000 being positive.

**The practical implication:**
The loss CAN still decrease in the negative-curvature directions from a saddle point — you're not actually stuck at a minimum. The challenge is that the gradient is zero at the saddle point, making it look like a local minimum to gradient descent. But a small perturbation (from SGD noise) reveals the downward direction.

**Part 2:**
**Saddle points as the real enemy:**

**What saddle points are:**
Critical points where ∇L = 0 but the Hessian has both positive and negative eigenvalues. From the saddle:
- In positive-curvature directions: the function curves upward (like a local minimum)
- In negative-curvature directions: the function curves downward (loss can still decrease)

**Why they are dangerous:**
Near a saddle point, the gradient is very small (approaching zero as you approach the critical point). Gradient descent slows dramatically — the update θ ← θ - η∇L becomes θ ← θ - η×0 = θ. The optimizer appears to have converged but is actually stuck on a plateau where the loss is far from minimal.

In high dimensions, saddle points can have very flat regions around them (the "plateau" problem) — the optimizer may spend many iterations making negligible progress in the flat region before finally escaping.

**How mini-batch SGD escapes:**
Each mini-batch provides a noisy gradient estimate g_batch ≠ ∇L_true even at the saddle point. The noise has components in all D dimensions, including the negative-curvature dimensions. This noise acts as a perturbation that kicks the optimizer off the plateau and into the downhill directions. The stochasticity effectively provides a random search that can find the escape directions.

**How Momentum helps:**
After escaping a saddle point, the optimizer enters a downhill direction. Momentum accumulates velocity in this direction, accelerating escape from the saddle point's neighborhood. Without momentum, the small gradient near the saddle causes very slow progress even after escaping; with momentum, the accumulated velocity carries the optimizer quickly away.

**Part 3:**
**Sharp vs. Flat Minima:**

**Sharp minimum:** The loss function drops into a narrow canyon — the loss increases steeply in all directions from the minimum. Small perturbation to the parameters → large increase in loss.

**Flat minimum:** The loss function sits in a wide, shallow basin — the loss changes very little for substantial parameter perturbations. Large region of parameter space all achieves approximately the same low training loss.

**Why flat minima generalize better:**
Generalization requires the model to perform well on TEST data, which has a slightly different distribution than training data. Testing introduces a small "shift" in the effective loss landscape.

For a **sharp minimum:** The test loss landscape is slightly shifted from training loss landscape. The sharp walls of the training minimum now position the model on the steep canyon walls of the test loss — test loss is high even though training loss is low.

For a **flat minimum:** The test loss landscape is similarly shifted, but the wide basin means the model is still near the bottom of the test loss basin. The flat walls are forgiving — small shifts in the landscape don't move you far from the minimum.

**Why SGD+Momentum finds flat minima:**
SGD's mini-batch noise causes the optimizer to "bounce around" the loss landscape rather than immediately descending to the nearest minimum. Sharp minima have small, narrow basins — the probability of the noisy SGD trajectory landing in a narrow canyon is low, and the steep walls cause large gradient oscillations that bounce the optimizer back out.

Wide, flat minima have large basins — SGD is more likely to wander into them and stay, because the gentle walls don't generate large opposing gradients. SGD effectively has an implicit preference for flat minima due to its noise characteristics.

**Why Adam might find sharp minima:**
Adam's adaptive per-parameter learning rates make its steps very "precise" and efficient — it homes in on the nearest minimum very rapidly. This rapid, directed descent is more likely to plunge into the first minimum encountered — which may be a narrow, sharp minimum with poor generalization properties. Adam doesn't bounce around enough to explore the landscape and find the wider basins.

This is why SGD+Momentum often achieves better test accuracy on image classification despite Adam converging faster in terms of training loss — the flat minima found by SGD generalize better to test data.

---

## BONUS CHALLENGE QUESTIONS

---

**Question 11**
Cross-topic synthesis.

1. You are training a logistic regression model on a dataset with N=500, D=5 (small problem) vs. N=10,000,000, D=100,000 (massive problem). For each scenario, specify: which optimization algorithm to use (IRLS, Batch GD, Mini-Batch SGD), which learning rate schedule (if any), and whether adaptive optimizer (Adam, RMSProp) would help. Justify every choice.
2. Explain the evolution of optimizers: GD → SGD → SGD+Momentum → AdaGrad → RMSProp → Adam. For each transition, state what specific failure mode was being fixed and what the fix introduced. This should be a coherent narrative.
3. A colleague claims: "Adam always converges faster than SGD, so we should always use Adam." Write a complete response addressing: (a) when this claim is true, (b) when SGD+Momentum is actually better, (c) the role of sharp vs. flat minima, (d) practical recommendations for different problem types.

### ✅ Answer

**Part 1:**
**Algorithm selection for two scales:**

**Small problem (N=500, D=5, logistic regression):**

**Algorithm:** IRLS

Reasoning:
- D=5 → Hessian is 5×5 = 25 entries — trivial to compute and invert (O(5³) = 125 operations per inversion)
- N=500 → gradient computation O(N×D) = 2,500 operations per step — very fast
- IRLS converges in ~10-20 iterations (quadratic convergence) vs. thousands for gradient descent
- No learning rate hyperparameter to tune — critical advantage for small problems where you want reliable, reproducible results
- Strong theoretical guarantees matter for statistical inference applications

**Learning rate schedule:** None needed — IRLS automatically determines optimal step sizes.

**Adaptive optimizer (Adam/RMSProp):** Unnecessary — IRLS is already second-order adaptive. Using Adam here would be like putting bicycle training wheels on a Formula 1 car.

**Large problem (N=10,000,000, D=100,000, logistic regression):**

**Algorithm:** Mini-Batch SGD (B=256-1024)

Reasoning:
- IRLS: Hessian is 100,000×100,000 = 10¹⁰ entries → 80 GB memory just for matrix storage → completely infeasible
- Batch GD: O(N×D) = O(10¹²) operations per step → hours per single update → completely impractical
- Mini-Batch SGD: O(B×D) = O(100,000×256) = O(2.56×10⁷) per step → milliseconds → feasible

**Learning rate schedule:** Cosine Annealing (or Step Decay as simpler alternative)
- Start with larger η (0.01-0.1) for rapid initial progress through the 10M example dataset
- Decay toward smaller η for precise convergence
- With 10M examples and reasonable batch size, each "epoch" takes thousands of steps — scheduling over epochs is practical

**Adaptive optimizer:** Yes, Adam recommended

Reasoning for logistic regression specifically:
- Logistic regression loss IS convex → Adam will find the global minimum (no sharp/flat minima concern)
- D=100,000 features likely includes many sparse and rarely-updated features → Adam's per-parameter adaptive rates are especially valuable
- Reduced need for learning rate tuning — important at this scale where tuning is expensive
- Note: For neural networks (non-convex), SGD+Momentum might be preferred; for convex logistic regression, Adam's advantages dominate

**Part 2:**
**Evolution narrative — fixing failure modes:**

**1. Batch Gradient Descent (BGD):**
The baseline. Uses all N examples per update → exact gradient → stable convergence.

**Failure mode:** Computationally prohibitive for large N (N=10M → minutes per step). Cannot scale to modern datasets.

---

**2. SGD (fix: scale to large N):**
Use 1 random example per update → O(D) per step → can handle any N.

**New failure mode introduced:** Extremely noisy gradient estimates → unstable convergence → oscillates around minimum rather than converging → requires very small η and many iterations.

---

**3. SGD + Momentum (fix: reduce noise / accelerate consistent directions):**
Add velocity term that accumulates consistent gradient directions.

**What it fixed:** Oscillations across narrow valleys (ravines) — opposing gradients cancel in the oscillating dimension, consistent gradients accumulate in the descent direction. Also helps escape local minima and saddle points.

**New failure mode:** Single global learning rate η — all parameters move at the same speed. Doesn't adapt to different parameter sensitivities. Overshoots with accumulated momentum.

---

**4. AdaGrad (fix: per-parameter adaptive learning rates):**
Accumulate G_j = Σ(gradient_j)² → effective η_j = η/√G_j.

**What it fixed:** Different parameters need different learning rates. Rare features (small G_j) get large effective η; common features (large G_j) get small effective η. Excellent for sparse gradient problems.

**New failure mode:** G_j grows forever → effective learning rate → 0 → optimizer freezes after enough steps. Fatal for deep learning with long training runs.

---

**5. RMSProp (fix: don't accumulate gradients forever):**
Replace sum with exponential moving average: v_j = β v_j + (1-β)g²_j.

**What it fixed:** AdaGrad's freezing problem. Old gradients are forgotten → v_j stabilizes at recent gradient magnitude → effective learning rate remains non-zero throughout training.

**New failure mode:** Still single estimate of gradient scale — doesn't use gradient direction (momentum). Also biased toward zero at initialization.

---

**6. Adam (fix: combine momentum with adaptive rates + fix initialization bias):**
Track both first moment (momentum) and second moment (RMSProp), with bias correction.

**What it fixed:**
- No momentum in RMSProp — Adam adds first-moment tracking for directional acceleration
- Initialization bias — bias correction ensures proper scale in early iterations
- Combined effect: adaptive per-parameter rates AND acceleration in consistent directions

**Remaining trade-off:** May converge to sharp minima in non-convex settings; may not generalize as well as SGD on some problems. The story continues...

---

**Part 3:**
**Complete response to "Adam always beats SGD":**

**(a) When Adam's faster convergence is actually true:**

1. **Non-stationary and sparse problems:** NLP, recommendation systems, text classification — features have very different update frequencies. Adam's per-parameter adaptive rates provide genuine advantage. Adam converges faster and to better solutions.

2. **Early training phase:** Across essentially all problem types, Adam reaches a reasonable loss level much faster than SGD — the combination of momentum and adaptive rates is powerful in the initial, noisy phase of training.

3. **Neural networks for NLP tasks:** BERT, GPT, transformers — these train faster and better with Adam. This is an empirical consensus.

4. **Fine-tuning pretrained models:** The adaptive rates help navigate the complex, highly non-convex landscape of an already partially-trained model.

**(b) When SGD+Momentum is actually better:**

1. **Computer vision CNNs:** ResNet, VGG, and similar architectures consistently achieve better test accuracy with SGD+Momentum (with careful tuning) than Adam. This has been demonstrated in numerous papers and competitions.

2. **When final generalization matters more than training speed:** If you have enough compute budget and care about last-1-2% of accuracy, SGD often wins.

3. **Overparameterized models:** When the model has many more parameters than data points, the implicit regularization of SGD's noise becomes crucial.

**(c) The sharp vs. flat minima role:**

Adam's rapid, adaptive steps efficiently navigate toward the nearest minimum. In non-convex landscapes with many minima of different quality:
- Adam tends to find narrow, sharp minima because its adaptive rates allow it to descend precisely into any nearby basin — including narrow ones
- Sharp minima generalize poorly: small perturbation from train → test distribution shifts the model to the steep walls of the canyon → high test loss

SGD+Momentum with its noisier, less-efficient trajectory:
- Bounces out of narrow minima (steep walls create large oscillations that escape)
- Settles in wide, flat basins (gentle walls don't create large opposing forces)
- Flat minima generalize well: test distribution shift keeps model near the basin center

This is the fundamental reason for Adam's worse generalization despite better training convergence — it optimizes training loss too well, landing in the wrong type of minimum.

**(d) Practical recommendations:**

| Problem Type | Recommendation | Reason |
|-------------|----------------|--------|
| Deep learning, NLP, transformers | Adam (β₁=0.9, β₂=0.999) | Sparse gradients, fast convergence important |
| CV/image classification | SGD+Momentum (with cosine annealing) | Better generalization, flat minima |
| Logistic regression, linear models | IRLS (small) or Adam (large scale) | Convex → no sharp/flat issue; Adam handles sparsity |
| Neural networks, production | Try both, validate on held-out test set | Task-dependent; empirical comparison essential |
| Research/state-of-the-art models | SGD+Momentum with tuned LR schedule | Often last % of accuracy comes from SGD |
| Rapid prototyping | Adam | Works "out of the box" without extensive tuning |

**The core message:**
Adam is better at reaching good training loss quickly with minimal tuning. SGD+Momentum is better at finding minima that generalize well, given sufficient tuning effort. The "right" answer is problem-specific and should always be validated empirically on a held-out test set — not decided theoretically.

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | Problem formulation, why closed-form fails | Medium | Very High |
| Q2 | Gradient descent fundamentals, update rule, requirements | Medium | Very High |
| Q3 | Learning rate effects, LR scheduling | Medium | Very High |
| Q4 | BGD vs SGD vs Mini-Batch comparison | Medium | Very High |
| Q5 | Newton's method, IRLS algorithm | Hard | High |
| Q6 | IRLS vs GD comparison, when to use each | Hard | High |
| Q7 | SGD limitations, Momentum, physics analogy | Medium-Hard | Very High |
| Q8 | AdaGrad, RMSProp, Adam with bias correction | Hard | Very High |
| Q9 | NAG vs Momentum, lookahead concept | Hard | Medium |
| Q10 | Non-convex optimization, sharp/flat minima | Very Hard | High |
| Q11 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 7 Most Likely Exam Questions From This Topic

1. **Batch GD vs SGD vs Mini-Batch** — explain conceptual difference, pros/cons of each (mirrors Q5 from original exam exactly)
2. **Momentum** — what it does, physics analogy, how it helps escape suboptimal regions (mirrors Q5 part 3 from original exam)
3. **Adam** — what it combines, why popular, when it's NOT better than SGD
4. **Learning rate effects** — too large diverges, too small slow, just right converges; learning rate scheduling
5. **Why closed-form fails for logistic regression** — gradient is nonlinear in w, contrast with linear regression
6. **IRLS vs Gradient Descent** — comparison table, when to use each, computational costs
7. **Sharp vs flat minima** — why flat generalizes better, why SGD finds flat minima

**Send the next slides and I will continue building the complete exam for every topic!**