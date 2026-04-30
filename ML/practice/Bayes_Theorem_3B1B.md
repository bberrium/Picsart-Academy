# Bayes' Theorem — The Geometry of Changing Beliefs
> **Source:** [3Blue1Brown — Grant Sanderson (Dec 22, 2019)](https://youtu.be/HZGCoVF3YvM)
> **Companion:** [The Quick Proof of Bayes' Theorem](https://youtu.be/U_85TaXbeIo) · [The Medical Test Paradox](https://youtu.be/lG4VkPoG3ko)
> **Lesson page:** [3blue1brown.com/lessons/bayes-theorem](https://www.3blue1brown.com/lessons/bayes-theorem)

---

## Why This Matters

Bayes' theorem is described as one of the most important formulas in all of probability. Its real-world applications include:

- **Scientific discovery** — quantifying how new data validates or invalidates models
- **Machine learning & AI** — explicitly modeling a machine's belief state
- **Treasure hunting** — in the 1980s, a team led by Tommy Thompson used Bayesian search tactics to locate a ship that sank 150 years earlier, carrying ~$700 million in gold
- **Medicine** — interpreting test results correctly (see companion video)
- **Epistemology** — reframing how you think about updating your own beliefs

---

## Three Levels of Understanding

Grant structures the video around three depths of understanding, taught in **reverse order** (hardest first):

| Level | Description |
|-------|-------------|
| 3 — Recognizing | Knowing **when** to use Bayes' theorem |
| 2 — Why it's true | Understanding the geometric/logical proof |
| 1 — Plugging in | Just knowing what each symbol means |

> The most important level is 3 — recognizing when Bayes applies in the first place.

---

## 1 — The Motivating Example: Steve the Librarian

### The setup (from Kahneman & Tversky)

> *"Steve is very shy and withdrawn, invariably helpful but with very little interest in people or the world of reality. A meek and tidy soul, he has a need for order and structure, and a passion for detail."*

**Question:** Is Steve more likely to be a librarian or a farmer?

This example is taken from research by psychologists **Daniel Kahneman** and **Amos Tversky**, whose work won the Nobel Prize and was popularized in books like *Thinking Fast and Slow* (Kahneman) and *The Undoing Project* (Michael Lewis).

### What most people do

Most people say "librarian" — because Steve's description aligns with the *stereotype* of a librarian.

### What the research shows is wrong

People fail to account for the **base rate** — the ratio of farmers to librarians in the general population.

In the US, that ratio is approximately **20 to 1** (farmers to librarians). Even if the personality description is used, this massive prior imbalance must be factored in.

> **Key insight:** Rationality is not about knowing facts. It's about recognizing *which* facts are relevant.

---

## 2 — Reasoning Through It (The Heart of Bayes')

### Representative sample approach

Instead of abstract math, picture a concrete sample of people:

```
Population sample:
┌───────────────────────────────────────────┐
│  200 Farmers                              │
│  10 Librarians                            │
│  Total: 210 people                        │
└───────────────────────────────────────────┘
```

Now apply your gut estimates about the personality description:
- **40%** of librarians fit the "meek and tidy" description
- **10%** of farmers fit the description

```
Who fits the description?

  Librarians:  40% × 10  =  4 people
  Farmers:     10% × 200 = 20 people
  ─────────────────────────────────────
  Total fitting description:    24 people
```

### The answer

$$P(\text{librarian} \mid \text{fits description}) = \frac{4}{24} \approx 16.7\%$$

Even though librarians are **4× more likely** than farmers to fit the description, the huge prior imbalance (20:1 ratio of farmers) means Steve is still much more likely to be a farmer.

### The mantra

> **New evidence should not determine your beliefs in a vacuum. It should update prior beliefs.**

Before reading the description: ~1/21 chance Steve is a librarian.
After reading the description: update to 4/24 ≈ 1/6 chance.
The evidence shifted beliefs significantly — but did not override the prior.

---

## 3 — Generalizing to the Formula

### Setting up the notation

The general situation:
- **H** = Hypothesis (e.g., "Steve is a librarian")
- **E** = Evidence (e.g., "Steve fits the meek/tidy description")
- Goal: find $P(H \mid E)$ — probability hypothesis is true *given* the evidence

The vertical bar " | " means "given that" — we are **restricting** our view to possibilities where the evidence holds.

### Naming the pieces

| Term | Symbol | Meaning | In our example |
|------|--------|---------|----------------|
| **Prior** | $P(H)$ | Probability of hypothesis before seeing evidence | 10/210 = 1/21 |
| **Likelihood** | $P(E \mid H)$ | Probability of seeing evidence *if* hypothesis is true | 40% = 0.4 |
| **False positive rate** | $P(E \mid \neg H)$ | Probability of seeing evidence *if* hypothesis is false | 10% = 0.1 |
| **Posterior** | $P(H \mid E)$ | Probability of hypothesis *after* seeing evidence | 4/24 ≈ 16.7% |

The ¬ (elbow symbol) means "not."

### Deriving the formula

The numerator: how many people are librarians **and** fit the description?
$$\text{numerator} = \underbrace{N \cdot P(H)}_{\text{total librarians}} \times P(E \mid H) = N \cdot P(H) \cdot P(E \mid H)$$

The denominator: how many people **total** fit the description?
$$\text{denominator} = N \cdot P(H) \cdot P(E \mid H) + N \cdot P(\neg H) \cdot P(E \mid \neg H)$$

The $N$ (arbitrary sample size = 210) cancels out entirely, leaving:
![[Pasted image 20260425193948.png]]

$$\boxed{P(H \mid E) = \frac{P(H) \cdot P(E \mid H)}{P(H) \cdot P(E \mid H) + P(\neg H) \cdot P(E \mid \neg H)}}$$

This is **Bayes' Theorem**.

The denominator is often written compactly as $P(E)$, the total probability of seeing the evidence:

$$P(H \mid E) = \frac{P(H) \cdot P(E \mid H)}{P(E)}$$

But in practice, $P(E)$ almost always needs to be expanded into the two-case form above.

---

## 4 — The Geometry: The Rectangle Diagram

Grant strongly recommends **not memorizing the formula** — instead, draw this diagram on the fly.

### How to build it

**Step 1:** Draw a 1×1 square. The total area = 1 = all possibilities.

```
┌────────────────────────────────┐
│                                │
│                                │  Total area = 1
│                                │
└────────────────────────────────┘
```

**Step 2:** Divide the width at $P(H)$ — left side = hypothesis true, right side = hypothesis false.

```
←P(H)→←────P(¬H)────────────→
┌──────┬──────────────────────┐
│      │                      │
│  H   │        ¬H            │
│      │                      │
└──────┴──────────────────────┘
```

**Step 3:** From the left column, shade the fraction $P(E \mid H)$ — this is the area where both H is true **and** E is observed.

**Step 4:** From the right column, shade the fraction $P(E \mid \neg H)$ — this is the area where H is false but E is still observed (false positives).

```
←P(H)→←────P(¬H)────────────→
┌──────┬──────────────────────┐
│██████│███                   │  ← shaded = evidence E is present
│██████│███                   │
│      │                      │
└──────┴──────────────────────┘
 ↑left    ↑right
 rectangle rectangle
 area =   area =
 P(H)·    P(¬H)·
 P(E|H)   P(E|¬H)
```

**Step 5:** The posterior $P(H \mid E)$ = area of left shaded rectangle ÷ total shaded area.

```
                left shaded area
P(H|E)  =  ──────────────────────────
            left shaded + right shaded
```

### Why this works

- When $P(E \mid H) = P(E \mid \neg H)$: both rectangles are the same height → shading is proportional to width → posterior = prior → **irrelevant evidence changes nothing**
- When likelihoods differ greatly: one rectangle is much taller → prior gets updated significantly

The diagram makes it visually obvious: Bayes' theorem is just asking "what fraction of the shaded area is on the left?"
![[Pasted image 20260425194036.png]]

---

## 5 — Probability as Proportions

A key reframe from the video:

> **Probability is not really the study of uncertainty — it's the math of proportions.**

All of Bayes' theorem can be read as a statement about proportions:

- Look at the subset of cases where the evidence is true
- Within that subset, find the fraction where the hypothesis is also true
- That fraction is the posterior

The right-hand side of the formula just spells out how to compute this — it's actually *obvious* once you see it that way.

### Why geometry beats numbers

| Method | Pros | Cons |
|--------|------|------|
| Representative sample (e.g., 210 people) | Intuitive, concrete | Doesn't capture continuous probability |
| Area / geometry | Continuous, flexible, sketchable | Slightly more abstract |
| Abstract formula | Compact, general | Opaque without context |

---

## 6 — The Linda Problem (Conjunction Fallacy)

Grant introduces a second Kahneman & Tversky result to illustrate how framing affects reasoning:

### The problem

> *"Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations."*

**Question:** What is more likely?
1. Linda is a bank teller
2. Linda is a bank teller **and** is active in the feminist movement

**85% of participants** chose option 2 — even though it's logically impossible for a subset to be larger than the full set.

```
All bank tellers
┌────────────────────────────┐
│   bank tellers who are     │
│   feminist activists  ████ │
│                            │
└────────────────────────────┘
The subset CANNOT be larger than the whole set.
```

### The fix: use counts, not probabilities

When participants were asked instead:
> "There are 100 people who fit Linda's description. How many are bank tellers? How many are bank tellers who are feminist activists?"

**Nobody made the error.** Everybody correctly answered that the bank teller count must be higher.

### Why this happens

> Phrases like "40 out of 100" kick our intuitions into gear much more effectively than "40%", much less "0.4", much less abstract references to likelihood.

This is why Grant recommends thinking in representative samples or areas — they engage the right cognitive machinery.

---

## 7 — What Shifts in the Diagram

The diagram doubles as a tool for thinking about contested questions:

| If you debate... | ...adjust this in the diagram |
|-----------------|-------------------------------|
| Context (who is Steve exactly? Randomly sampled? A friend?) | **Prior** — shifts the left/right split |
| Personality stereotypes about librarians and farmers | **Likelihoods** — changes the height of the shaded rectangles |

Grant acknowledges that some psychologists dispute Kahneman & Tversky's conclusion. The context of the question genuinely changes what the "rational" prior should be. But:

> **Whether or not you accept this particular experiment, the principle that evidence should update beliefs — not replace them — is worth tattooing in your brain.**

---

## 8 — The Quick Proof (Companion Video)

The companion video ([YouTube](https://youtu.be/U_85TaXbeIo)) derives Bayes' theorem from the definition of **conditional probability**.

### Definition of conditional probability

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

This says: restrict to outcomes where B occurred; within those, what fraction also have A?

### Deriving Bayes' theorem in two lines

From the definition of conditional probability, applied symmetrically:

$$P(H \text{ and } E) = P(H) \cdot P(E \mid H)$$
$$P(H \text{ and } E) = P(E) \cdot P(H \mid E)$$

Since both expressions equal $P(H \text{ and } E)$, set them equal and solve:

$$P(E) \cdot P(H \mid E) = P(H) \cdot P(E \mid H)$$

$$\boxed{P(H \mid E) = \frac{P(H) \cdot P(E \mid H)}{P(E)}}$$

### Independence (common misconception)

Two events A and B are **independent** if:
$$P(A \mid B) = P(A)$$

Knowing B happened tells you nothing about A. This is equivalent to:
$$P(A \text{ and } B) = P(A) \cdot P(B)$$

**Common misconception:** people confuse mutually exclusive with independent.
- **Mutually exclusive**: A and B cannot both happen → $P(A \text{ and } B) = 0$
- **Independent**: knowing one happens gives no information about the other

These are almost *opposite*: if A and B are mutually exclusive (and have non-zero probability), knowing A happened tells you definitively that B did *not* — so they are highly dependent.

---

## 9 — Applications & Broader Significance

### Scientific inference

Scientists use Bayes' theorem to quantify how much new experimental data validates or invalidates a hypothesis. The prior is the existing state of knowledge; the posterior is the updated belief after the data.

$$P(\text{model} \mid \text{data}) = \frac{P(\text{data} \mid \text{model}) \cdot P(\text{model})}{P(\text{data})}$$

### Machine learning / AI

In Bayesian classifiers and probabilistic models, the machine maintains explicit numerical beliefs about states of the world and updates them via Bayes' rule as new observations arrive.

### Medical testing paradox

A positive test ≠ having the disease. Even a highly accurate test can produce mostly false positives if the disease is rare. This is the subject of the companion video [The Medical Test Paradox](https://youtu.be/lG4VkPoG3ko).

Example structure (not from video, for reference):
- Disease prevalence: 1%
- Test sensitivity: 98%
- Test specificity: 94%
- P(disease | positive test) = much lower than intuition suggests

### Epistemology

Bayes' theorem offers a mathematical model for rational belief change:
- You have a prior (what you believed before)
- You observe evidence
- You compute a posterior (updated belief)
- The posterior becomes the prior for the next round

This iterative process is called **Bayesian updating** and it formalizes what it means to "change your mind rationally."

---

## 10 — Formula Reference & Cheatsheet

### Bayes' Theorem

$$P(H \mid E) = \frac{P(H) \cdot P(E \mid H)}{P(H) \cdot P(E \mid H) + P(\neg H) \cdot P(E \mid \neg H)}$$

Or compactly:

$$P(H \mid E) = \frac{P(H) \cdot P(E \mid H)}{P(E)}$$

### Named terms

| Name | Symbol | Role |
|------|--------|------|
| Prior | $P(H)$ | Belief before evidence |
| Likelihood | $P(E \mid H)$ | How probable is evidence if H is true? |
| Marginal likelihood | $P(E)$ | Total probability of evidence |
| Posterior | $P(H \mid E)$ | Belief after evidence |

### Conditional probability (foundation)

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

### Independence

$$A \perp B \iff P(A \mid B) = P(A) \iff P(A \text{ and } B) = P(A) \cdot P(B)$$

### The Steve example worked out

| Quantity | Value |
|----------|-------|
| $P(H)$ = P(librarian) | 10/210 ≈ 0.048 |
| $P(\neg H)$ = P(farmer) | 200/210 ≈ 0.952 |
| $P(E \mid H)$ = P(description | librarian) | 0.40 |
| $P(E \mid \neg H)$ = P(description | farmer) | 0.10 |
| Numerator = $P(H) \cdot P(E \mid H)$ | 0.048 × 0.40 ≈ 0.019 |
| Denominator = $P(E)$ | 0.019 + (0.952 × 0.10) = 0.019 + 0.095 = 0.114 |
| **Posterior** $P(H \mid E)$ | 0.019 / 0.114 ≈ **16.7%** |

---

## 11 — How to Use the Diagram (Step-by-Step)

When faced with any Bayes problem, draw this instead of remembering the formula:

```
Step 1: Draw a 1×1 square
Step 2: Mark P(H) on the x-axis → split into left (H) and right (¬H)
Step 3: Shade height P(E|H) from the left column
Step 4: Shade height P(E|¬H) from the right column
Step 5: Answer = left shaded area / total shaded area

           P(H)        P(¬H)
        ┌────────┬─────────────────┐
 P(E|H) │████████│                 │
        │████████│─────────────────│ P(E|¬H)
        │        │█████            │
        │        │█████            │
        └────────┴─────────────────┘

  Answer = area of left shading / (left + right shading)
         = P(H)·P(E|H) / [P(H)·P(E|H) + P(¬H)·P(E|¬H)]
```

The diagram works for any values. No formula memorization required.

---

## 12 — Related Concepts & Videos

```
Bayes' Theorem (this video)
       │
       ├─→ The Quick Proof of Bayes' Theorem
       │   https://youtu.be/U_85TaXbeIo
       │
       ├─→ The Medical Test Paradox (redesigning Bayes' rule via Likelihood Ratios)
       │   https://youtu.be/lG4VkPoG3ko
       │
       └─→ But What Is the Central Limit Theorem?
           https://youtu.be/zeJD6dqJ5lo
```

### Source material cited in lesson page

- **Kahneman, D.** — *Thinking, Fast and Slow* (the Steve and Linda examples)
- **Lewis, M.** — *The Undoing Project* (popularization of Kahneman-Tversky work)
- **Kahneman & Tversky original research** — the representativeness heuristic papers

---

## 13 — Obsidian Links

- [[Conditional Probability]] — foundation; P(A|B) = P(A and B)/P(B)
- [[Prior and Posterior]] — belief before and after evidence
- [[Likelihood]] — P(evidence | hypothesis)
- [[Base Rate Neglect]] — the cognitive error the Steve example illustrates
- [[Conjunction Fallacy]] — the Linda problem; P(A and B) ≤ P(A)
- [[Independence]] — P(A|B) = P(A); not the same as mutually exclusive
- [[Bayesian Inference]] — the full framework built on Bayes' theorem
- [[Law of Total Probability]] — how to compute P(E) from its parts
- [[Central Limit Theorem 3B1B]] — companion video in the probability series
- [[Normal Distribution]] — related probability topic

---

*Notes compiled from the full video transcript, the 3Blue1Brown lesson page, and the companion quick proof video.*
*Main video: [youtube.com/watch?v=HZGCoVF3YvM](https://www.youtube.com/watch?v=HZGCoVF3YvM)*
*Quick proof: [youtube.com/watch?v=U_85TaXbeIo](https://www.youtube.com/watch?v=U_85TaXbeIo)*
