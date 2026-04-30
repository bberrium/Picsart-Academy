# But What Is the Central Limit Theorem?
> **Source:** [3Blue1Brown — Grant Sanderson (Mar 14, 2023)](https://youtu.be/zeJD6dqJ5lo)
> **Series:** Probability → CLT → [Why π is in the Normal Distribution](https://www.3blue1brown.com/lessons/gaussian-integral) → [Convolutions](https://www.youtube.com/watch?v=KuXjwB4LzSA)

---

## Video Structure (Timestamps)

| Time | Chapter |
|------|---------|
| 0:00 | Introduction |
| 1:53 | A Simplified Galton Board |
| 4:14 | The General Idea |
| 6:15 | Dice Simulations |
| 8:55 | True Distributions for Sums |
| 11:41 | Mean, Variance, and Standard Deviation |
| 15:54 | Unpacking the Gaussian Formula |
| 20:47 | The More Elegant Formulation |
| 25:01 | A Concrete Example |
| 27:10 | Sample Means |
| 28:10 | Underlying Assumptions |


---

## 1 — Introduction: The Galton Board

### What it shows
A **Galton board** (also called a bean machine or quincunx) drops balls through rows of pegs. Each peg deflects a ball randomly left or right. Despite each individual deflection being chaotic and unpredictable, the aggregate outcome is remarkably predictable: balls accumulate in a **bell-curve shape**.

```
        ●
       / \
      ●   ●
     / \ / \
    ●   ●   ●
   / \ / \ / \
  ●   ●   ●   ●
 [  ][  ][  ][  ]   ← buckets fill as a bell curve
```

### Why this is remarkable
- Individual event: **unknowable**
- Large-number aggregate: **precisely predictable**
- The distribution that appears is called the **Normal distribution** (= bell curve = Gaussian distribution)

### Normal distributions are everywhere
- Heights of people in the same demographic → normal distribution
- Number of distinct prime factors of large natural numbers → normal distribution
- Measurement errors in experiments → normal distribution

### The core question
> *Why is the normal distribution so common?*

The answer is the **Central Limit Theorem** — one of the crown jewels of probability theory.

---

## 2 — A Simplified Galton Board Model

### The idealized model (not physically accurate)
Grant strips away real physics and uses a pure mathematical model to isolate the CLT cleanly:

1. Ball hits a central peg
2. **50/50 chance**: bounces left (−1) or right (+1)
3. Falls onto next peg — same 50/50 choice
4. Repeat for **n rows** of pegs
5. Final position = **sum of n choices** (each ±1)

```
Visual: 5-row board

Row 1:        [peg]
             /     \
Row 2:    [peg]   [peg]
           / \     / \
Row 3: [p] [p]   [p] [p]
           ...
Buckets: -5  -3  -1  +1  +3  +5
         (labeled by their sum)
```

With 5 rows, each ball makes 5 binary choices. The final bucket = the sum of those choices, e.g., (+1, −1, +1, +1, −1) → sum = +1.

### Connection to Pascal's Triangle
The number of paths to each bucket follows Pascal's triangle. For fair 50/50 bounces, the probabilities are the binomial coefficients.

```
Row 5 probabilities (normalized):
Bucket:  -5   -3   -1   +1   +3   +5
Count:    1    5   10   10    5    1
Prob:   1/32 5/32 10/32 10/32 5/32 1/32
```

This forms the **beginning** of a bell-curve shape, even with just 5 rows.

---

## 3 — The General Idea (CLT in Plain Language)

### Key vocabulary

> **Random Variable X** — a random process where each outcome is associated with a number.

Examples:
- One peg bounce: outcomes {−1, +1}, each with probability 1/2
- One die roll: outcomes {1, 2, 3, 4, 5, 6}, each with probability 1/6

### The setup
- Take **n independent samples** of X: $X_1, X_2, \ldots, X_n$
- Form their **sum**: $S_n = X_1 + X_2 + \cdots + X_n$
- Ask: *what does the distribution of $S_n$ look like?*

### The claim

> **Central Limit Theorem (informal):** As n → ∞, the distribution of $S_n$ looks more and more like a bell curve, regardless of the distribution of the individual variable X.

```
n = 2:   distribution of sum ≈ original shape (skewed, lumpy)
n = 5:   more symmetric, lumps start smoothing
n = 10:  clearly bell-shaped
n = 50:  essentially perfect bell curve
n = ∞:   exactly a normal distribution
```

### Example question (goal for the video)
> Roll a die 100 times. Add the results. Find the smallest range of values such that you are **95% certain** the sum falls within it.

You can answer this for a **fair** or a **weighted** die — CLT handles both.

---

## 4 — Dice Simulations

### Simulation setup
- Start with a **non-uniform (weighted) die** — distribution skewed toward low values
- Take **10 samples** from this distribution
- Record the **sum** of those 10 samples
- Repeat thousands of times and plot the sums

### What you observe

| Dice in sum | Shape of sum distribution |
|-------------|--------------------------|
| 2 | Resembles the original skewed shape |
| 5 | More symmetric, getting smoother |
| 10 | Looks like a bell curve (slight spikiness for extreme distributions) |
| 15 | Cleaner bell curve |

### Key insight from simulation
Even starting from a highly asymmetric (lopsided) distribution, adding more variables **washes away** all the original asymmetry. Symmetry **emerges from chaos**.

### Extreme distribution test
Distribution: nearly all probability at 1 and 6, almost none in between.

```
P(X):
  ████░░░░░░░░░░░░████
  1  2  3  4  5  6
```

With n=10: sum distribution is bell-shaped but **spiky** — 10 is not yet enough for this particular distribution. With n=50: perfectly smooth bell curve.

> ⚠️ How many samples you need before CLT "kicks in" depends on the original distribution. More extreme = more samples needed.

---

## 5 — True Distributions for Sums

### Moving from simulation to exact calculation

Simulations are illustrative but imprecise. For exact shapes, use **convolutions**.

#### Uniform die (each face prob = 1/6)
For a **pair of dice**, enumerate all 36 equally probable pairs and count how many add to each value:

```
Sum:   2   3   4   5   6   7   8   9  10  11  12
Count: 1   2   3   4   5   6   5   4   3   2   1
Prob: 1/36 ...                              1/36
```

Shape: a **triangle** (tent function). Adding a third die gives a smoother curve.

#### Non-uniform die
Same idea, but instead of counting pairs, **multiply** the probabilities:

$$P(S = k) = \sum_{j} P(X_1 = j) \cdot P(X_2 = k - j)$$

This operation is called a **convolution**. It is the weighted version of the counting game.

#### As n grows:

```
n=1:  original distribution shape
n=2:  convolution of distribution with itself → smoother
n=3:  convolve again → smoother still
...
n→∞: converges to bell curve
```

The video plots these exact distributions to show the bell curve emerging precisely, not just approximately.

---

## 6 — Mean, Variance, and Standard Deviation

As n grows, the sum distributions:
1. **Drift rightward** (mean grows)
2. **Spread wider** (standard deviation grows, but slower than mean)

Need tools to describe this quantitatively.

### Mean (μ)

> The **mean** = "center of mass" of the distribution = **expected value** of the random variable.

$$\mu = E[X] = \sum_{\text{all outcomes}} P(\text{outcome}) \times \text{value}$$

- Higher-value outcomes weighted more → higher mean
- Lower-value outcomes weighted more → lower mean

**How mean scales with n:**
$$E[X_1 + X_2 + \cdots + X_n] = n \cdot \mu$$

Each time we add another variable, mean increases by μ. Distributions march steadily rightward.

### Variance (σ²)

> **Variance** = expected squared deviation from the mean.

$$\sigma^2 = E[(X - \mu)^2] = \sum_{\text{outcomes}} P(\text{outcome}) \cdot (\text{value} - \mu)^2$$

Why square? Whether above or below mean, squaring gives a positive contribution. The further from the mean, the larger the penalty.

> ⚠️ Units of variance are **squared** (e.g., if X is in cm, variance is in cm²) — hard to interpret geometrically.

**Key property of variance:**
$$\text{Var}(X_1 + X_2 + \cdots + X_n) = n \cdot \sigma^2 \quad \text{(for independent variables)}$$

Variances **add** when variables are independent.

### Standard Deviation (σ)

> **Standard deviation** = square root of variance. Interpretable as a "typical distance" from the mean.

$$\sigma = \sqrt{\text{Var}(X)}$$

**How standard deviation of sum scales with n:**

$$\sigma_{S_n} = \sqrt{n} \cdot \sigma$$

Not n, but **√n**. This is crucial.

```
n=1:  σ
n=2:  √2 · σ  ≈ 1.41σ
n=4:  √4 · σ  = 2σ
n=9:  √9 · σ  = 3σ
n=100: 10σ
```

### The key tension

| Quantity | Grows as |
|----------|---------|
| Mean of sum | n · μ (linear) |
| Std dev of sum | √n · σ (sub-linear) |

The distribution spreads out, but **much more slowly** than it shifts. This is what makes the CLT useful for estimation: averaging many values concentrates probability mass near the true mean.

---

## 7 — Unpacking the Gaussian Formula

### Building the bell curve formula from scratch

**Step 1: Exponential decay in both directions**

Start with $e^{-x^2}$ — this decays symmetrically on both sides, giving the basic bell shape:

```
      *
    *   *
  *       *
*           *
         x
```

> Why not $e^{-|x|}$? That gives a sharp peak (cusp) at x=0. Squaring is smoother and mathematically tractable.

**Step 2: Add width control**

$$e^{-cx^2}$$

Increasing c → **narrower** bell. Decreasing c → **wider** bell.

Equivalently (by rules of exponents):

$$e^{-\frac{1}{2}\left(\frac{x}{\sigma}\right)^2}$$

When rewritten this way, **σ is the standard deviation** of the resulting distribution. Very readable.

> Note: The base e is not special for the shape — any positive base gives the same family of curves. We use e because it makes σ interpretable directly.

**Step 3: Normalize to area = 1**

This function must be a **probability density function** (PDF). For continuous distributions:

$$P(a \leq X \leq b) = \int_a^b f(x)\, dx$$

The **total area must = 1** (something must happen).

The area under $e^{-x^2}$ is **not 1** — it equals $\sqrt{\pi}$.

$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

> This is where π comes from! (The follow-up video explains the geometric reason via circles and convolutions.)

**Step 4: Full formula**

Divide by $\sqrt{\pi}$ to fix area, and by $\sigma\sqrt{2}$ to account for the width parameter:

$$\boxed{f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}}$$

| Parameter | Role |
|-----------|------|
| μ | Slides the curve left/right (sets the mean) |
| σ | Controls width (sets the standard deviation) |
| $\frac{1}{\sigma\sqrt{2\pi}}$ | Normalization constant to ensure area = 1 |

### Standard Normal Distribution

When μ = 0 and σ = 1:

$$f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

This is the **standard normal distribution**, written $\mathcal{N}(0, 1)$.

---

## 8 — The More Elegant Formulation

### Standardization / Z-score

Instead of tracking the raw sum $S_n$, track **how many standard deviations away from the mean** it is:

$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

This expression:
- **Subtracts** the expected mean → centers at zero
- **Divides** by the expected standard deviation → scales to unit variance

$Z_n$ always has mean 0 and standard deviation 1, regardless of n or the original distribution.

### The rigorous CLT statement

> For $Z_n$ as defined above, and for any real numbers $a < b$:
>
> $$\lim_{n \to \infty} P(a \leq Z_n \leq b) = \int_a^b \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\, dx$$

The right side is the area under the **standard normal curve** between a and b.

```
Standard Normal N(0,1):

         ____
       /      \
      /        \
-----/          \-----
    a            b
    |←——area——→|  ← this equals the probability
```

### The magical visual

When you:
1. Plot distributions of sums for n = 1, 2, 3, 5, 10, 50, ...
2. **Shift** each so the mean aligns at zero
3. **Rescale** each so the standard deviation = 1

→ All distributions converge to **one universal shape**: $\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$

It doesn't matter what distribution you started with. The original distribution's shape is **completely washed away**.

```
Start:     any shape (skewed, bimodal, U-shaped, uniform, ...)
n = 50+:   always → standard normal
```

---

## 9 — Concrete Example: 100 Dice Rolls

### Problem
Roll a **fair die** 100 times. Sum the results. Find the smallest interval such that there is a **95% chance** the sum falls within it.

### Step 1: Find μ (mean of one die)

$$\mu = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = \frac{21}{6} = 3.5$$

### Step 2: Find σ (std dev of one die)

First, the variance:

$$\sigma^2 = \frac{1}{6}\left[(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2\right]$$

$$= \frac{1}{6}[6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25] = \frac{17.5}{6} \approx 2.92$$

$$\sigma = \sqrt{2.92} \approx 1.71$$

### Step 3: Parameters of the sum S₁₀₀

$$\mu_{S_{100}} = 100 \times 3.5 = 350$$

$$\sigma_{S_{100}} = \sqrt{100} \times 1.71 = 10 \times 1.71 = 17.1$$

### Step 4: Apply the 68-95-99.7 Rule

For any normal distribution:

| Range | % of outcomes |
|-------|-------------|
| μ ± 1σ | ≈ 68% |
| **μ ± 2σ** | **≈ 95%** ← we want this |
| μ ± 3σ | ≈ 99.7% |

For **95%**, we need the interval μ ± 2σ:

$$350 \pm 2(17.1) = 350 \pm 34.2$$

$$\boxed{[315.8,\ 384.2]}$$

> You can be **95% confident** a sum of 100 fair dice rolls will fall between ~316 and ~384.

### Visual

```
       68%
    |←——————→|
    |←————————————→|
       95%
    |←——————————————————→|
           99.7%

    281   299   316   333  350  367   384   401   419
     |    |     |     |    |    |     |     |     |
   −3σ  −2σ   −1σ        μ        +1σ   +2σ   +3σ
```

---

## 10 — Sample Means

### Reframing: average instead of sum

The expression $Z_n$ divides by n and gets you the **empirical average** of the die rolls:

$$\bar{X}_n = \frac{S_n}{n} = \frac{X_1 + \cdots + X_n}{n}$$

For our example:

$$\mu_{\bar{X}_{100}} = 3.5 \quad \text{(same as original mean)}$$

$$\sigma_{\bar{X}_{100}} = \frac{\sigma}{\sqrt{n}} = \frac{1.71}{\sqrt{100}} = \frac{1.71}{10} = 0.171$$

The 95% interval for the **average** (not the sum) is:

$$3.5 \pm 2(0.171) = 3.5 \pm 0.342 = [3.158,\ 3.842]$$

### Key insight: Law of Large Numbers

The standard deviation of the sample mean shrinks as $1/\sqrt{n}$. Double the sample size → standard deviation divided by $\sqrt{2}$. Quadruple → halve the spread. This is **why averaging many measurements gives a precise estimate**.

| n | σ of average |
|---|-------------|
| 1 | 1.71 |
| 4 | 0.855 |
| 25 | 0.342 |
| 100 | 0.171 |
| 10,000 | 0.0171 |

---

## 11 — Underlying Assumptions (The Three Requirements)

The CLT only applies when **all three** of these hold.

### Assumption 1: Independence

Each variable $X_i$ must be **independent** of the others. The outcome of one does not influence any other.

✅ Fair dice: knowing one roll tells you nothing about the next.
❌ Galton board balls: each bounce changes the angle of incidence for the next peg.

### Assumption 2: Identical Distribution (IID)

All variables must be sampled from the **same distribution**. (Sometimes combined with Assumption 1 under the label **IID** — "independent and identically distributed.")

✅ Rolling the same die 100 times.
❌ Rolling a different die each time with different probabilities.

> Note: There are generalized versions of the CLT (Lyapunov CLT, Lindeberg CLT) that relax the "identically distributed" requirement under certain conditions.

### Assumption 3: Finite Variance

The underlying distribution must have a **finite variance** (equivalently, a well-defined, finite standard deviation).

✅ Almost all "practical" distributions (uniform, binomial, normal, Poisson, exponential, ...).
❌ **Cauchy distribution** — its variance is undefined (infinite). Averaging Cauchy-distributed values does not converge to a normal distribution no matter how large n gets.

```
Normal tails:  fall off extremely fast (like e^{-x²})
Cauchy tails:  fall off very slowly (like 1/x²)
               → variance integral diverges → CLT fails
```

> The Galton board actually **violates Assumptions 1 and 2**: consecutive bounces are not independent, and each peg presents a different distribution. Yet it still looks like a bell curve — suggesting there are deeper principles at work beyond the classical CLT.

---

## 12 — Summary: The Full Picture

```
Any distribution for X
(uniform, skewed, bimodal, U-shaped, ...)
       ↓
Take n i.i.d. samples and sum them
       ↓
Standardize: Z = (Sum − nμ) / (σ√n)
       ↓  (as n → ∞)
Standard Normal N(0,1)
= (1/√2π) · e^{−x²/2}
```

### What CLT gives you
| You know | You get |
|----------|---------|
| μ and σ of X | Full distribution of the sum for large n |
| Any starting shape | Same universal limiting shape |
| n samples | Std dev of sum = σ√n, std dev of mean = σ/√n |

### What CLT does NOT give you
- Exact distribution for finite n (only an approximation)
- Anything useful when variance is infinite (Cauchy, etc.)
- Results when variables are dependent (unless using extended CLTs)

---

## 13 — Key Formulas Cheatsheet

| Concept | Formula |
|---------|---------|
| Mean of X | $\mu = \sum P(x_i) \cdot x_i$ |
| Variance of X | $\sigma^2 = E[(X-\mu)^2]$ |
| Variance of sum | $\text{Var}(S_n) = n\sigma^2$ |
| Std dev of sum | $\sigma_{S_n} = \sigma\sqrt{n}$ |
| Mean of sum | $\mu_{S_n} = n\mu$ |
| Std dev of average | $\sigma_{\bar{X}} = \sigma/\sqrt{n}$ |
| Z-score | $Z = (S_n - n\mu)/(\sigma\sqrt{n})$ |
| Normal PDF | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| Standard normal | $f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ |
| 68-95-99.7 rule | μ ± σ / μ ± 2σ / μ ± 3σ |

---

## 14 — Related Videos & Next Steps

```
This video (CLT fundamentals)
       ↓
Why π is in the Normal Distribution
https://www.3blue1brown.com/lessons/gaussian-integral
       ↑
Convolutions (X+Y in probability)
https://youtu.be/KuXjwB4LzSA
```

- The follow-up explains **why** the CLT is true — why the Gaussian is the universal attractor
- Requires understanding **convolutions** and the **Fourier transform**
- The appearance of π connects to the geometry of circles in 2D

---

## 15 — Connections & Related Concepts

- [[Normal Distribution]] — the limiting shape
- [[Convolution]] — the operation underlying sums of random variables
- [[Law of Large Numbers]] — the average converges to μ
- [[Pascal's Triangle]] — counts paths in the simplified Galton board
- [[Probability Density Function]] — why area = probability for continuous distributions
- [[Variance]] — why it adds (not std dev); key to CLT's √n scaling
- [[IID]] — Independent and Identically Distributed assumption
- [[Cauchy Distribution]] — famous counterexample; CLT fails

---

*Notes compiled from the full video transcript and 3Blue1Brown lesson page.*
*Source code: [github.com/3b1b/videos/tree/master/_2023/clt](https://github.com/3b1b/videos/tree/master/_2023/clt)*
