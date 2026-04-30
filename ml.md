- [[#**Machine Learning**|**Machine Learning**]]
- [[#**Supervised Learning**|**Supervised Learning**]]
- [[#**Unsupervised Learning**|**Unsupervised Learning**]]
- [[#**Reinforcement Learning**|**Reinforcement Learning**]]
- [[#1. Linear Regression Basics|1. Linear Regression Basics]]
- [[#2. The Loss Function|2. The Loss Function]]
- [[#3. Finding the Best Parameters ($w_0$ and $w_1$)|3. Finding the Best Parameters ($w_0$ and $w_1$)]]
- [[#4. Gradient Descent|4. Gradient Descent]]
- [[#5. The Learning Rate ($\alpha$)|5. The Learning Rate ($\alpha$)]]
- [[#6. Types of Gradient Descent|6. Types of Gradient Descent]]
- [[#1. Classification Basics|1. Classification Basics]]
- [[#2. Building a Classifier (From Scratch to Linear)|2. Building a Classifier (From Scratch to Linear)]]
- [[#3. Logistic Regression|3. Logistic Regression]]
- [[#4. Loss Function and Optimization|4. Loss Function and Optimization]]
- [[#5. Derivatives for Gradient Descent|5. Derivatives for Gradient Descent]]
- [[#1. Multiclass Classification Overview|1. Multiclass Classification Overview]]
- [[#2. Multinomial Logistic Regression (MLR)|2. Multinomial Logistic Regression (MLR)]]
- [[#3. Multiclass Strategies|3. Multiclass Strategies]]
- [[#4. Multilabel Classification|4. Multilabel Classification]]
- [[#5: Data & Features|5: Data & Features]]
- [[#Metrics & Over/Underfitting|Metrics & Over/Underfitting]]
- [[#Metrics & Over/Underfitting#01. Data Splitting|01. Data Splitting]]
- [[#Metrics & Over/Underfitting#The Dataset Breakdown (Visual Diagram)|The Dataset Breakdown (Visual Diagram)]]
- [[#Metrics & Over/Underfitting#The Machine Learning Loop (Flowchart)|The Machine Learning Loop (Flowchart)]]
- [[#Metrics & Over/Underfitting#02. Cross-Validation|02. Cross-Validation]]
- [[#Metrics & Over/Underfitting#03. Metrics|03. Metrics]]
- [[#Metrics & Over/Underfitting#04. Overfitting & 05. Underfitting|04. Overfitting & 05. Underfitting]]
- [[#Metrics & Over/Underfitting#06. Bias and Variance|06. Bias and Variance]]
- [[#**KNN & Naive Bayes**|**KNN & Naive Bayes**]]
- [[#**KNN & Naive Bayes**#The Classification Problem|The Classification Problem]]
- [[#**KNN & Naive Bayes**#Identifying the Flaws (PlayTennis Example)|Identifying the Flaws (PlayTennis Example)]]
- [[#**KNN & Naive Bayes**#**What is KNN?**|**What is KNN?**]]
- [[#**KNN & Naive Bayes**#**Choosing 'K'**|**Choosing 'K'**]]
- [[#**KNN & Naive Bayes**#**KNN Pros & Cons**|**KNN Pros & Cons**]]
- [[#**KNN & Naive Bayes**#**Naive Bayes Pros & Cons**|**Naive Bayes Pros & Cons**]]
- [[#8: Support Vector Machine**|8: Support Vector Machine**]]
- [[#**Part 1: Support Vector Machine (SVM)**|**Part 1: Support Vector Machine (SVM)**]]
- [[#**Part 1: Support Vector Machine (SVM)**#Introduction to Thresholds (1D Example)|Introduction to Thresholds (1D Example)]]
- [[#**Part 1: Support Vector Machine (SVM)**#The Maximal Margin Classifier (MMC)|The Maximal Margin Classifier (MMC)]]
- [[#**Part 1: Support Vector Machine (SVM)**#The Soft Margin & Support Vectors|The Soft Margin & Support Vectors]]
- [[#**Part 1: Support Vector Machine (SVM)**#Moving to 2D (Height & Weight)|Moving to 2D (Height & Weight)]]
- [[#**Part 2: Kernel Trick**|**Part 2: Kernel Trick**]]
- [[#**Part 2: Kernel Trick**#Non-linear Separability|Non-linear Separability]]
- [[#**Part 2: Kernel Trick**#The Kernel Trick Explained|The Kernel Trick Explained]]
- [[#**Part 2: Kernel Trick**#Common Kernel Functions|Common Kernel Functions]]
- [[#**9. Decision Trees**|**9. Decision Trees**]]
- [[#**9. Decision Trees**#**01 Classification Trees**|**01 Classification Trees**]]
- [[#**9. Decision Trees**#**02 Entropy**|**02 Entropy**]]
- [[#**9. Decision Trees**#**03 Information Gain**|**03 Information Gain**]]
- [[#**9. Decision Trees**#**04 Pruning**|**04 Pruning**]]
- [[#**9. Decision Trees**#**05 Regression Trees**|**05 Regression Trees**]]
- [[#**9. Decision Trees**#**06 Random Forest**|**06 Random Forest**]]
- [[#**9. Decision Trees**#**1. Calculate Steps of Gradient Descent**|**1. Calculate Steps of Gradient Descent**]]
- [[#**9. Decision Trees**#**2. Calculate and Analyse Quality Metrics**|**2. Calculate and Analyse Quality Metrics**]]
- [[#**9. Decision Trees**#**3. Choose Algorithms & Preprocessing Based on the Task**|**3. Choose Algorithms & Preprocessing Based on the Task**]]
- [[#**9. Decision Trees**#**1. Example: Entropy & Information Gain**|**1. Example: Entropy & Information Gain**]]
- [[#**9. Decision Trees**#**2. Example: Gradient Descent (One Step)**|**2. Example: Gradient Descent (One Step)**]]
- [[#**9. Decision Trees**#**3. Example: Calculating Quality Metrics**|**3. Example: Calculating Quality Metrics**]]

1. Supervised Learning 
2. Linear Regression 
3. Linear Classifiers 
4. Multiclass Classification 
5. Multilabel Classification 
6. Gradient Descent and Variations 
7. Support Vector Machine 
8. K-Nearest Neighbors 
9. Naive Bayes 
10. 10.Decision Trees & Random Forest 
11. Cross-validation, Quality metrics 
12. 12.Bias and Variance, Under/Overfitting 
13. 13.Data & Features: Encoding and Scaling 


What should you be able to do (basics)? 
● Calculate steps of gradient descent
● Calculate and analyse quality metrics
● Choose algorithms, data preprocessing methods, etc based on the task description

---
### **Machine Learning**

- A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.
    

### **Supervised Learning**

- Each training example is a pair of <object, answer>.
    
- It is required to find the functional dependence of the answers on the descriptions of objects and construct an algorithm that takes the description of the object as an input and produces an answer at the output.
    

#### **Classification**

- The training set is a collection of individual objects $X=\{x_{i}\}i=1,...,n$ with the corresponding values $y_{i}$ T of the target variable for each object $X_{i},$ where T is a predetermined finite set of responses.
    
- In classification problems the set of valid answers is finite. They are called class labels.
    
- **Summary definition:** Answer is from finite set of possible outcomes.
    

#### **Regression**

- In regression problems, a continuous variable is predicted, the valid answer is a real number or a numeric vector.
    
- The training set is a collection of individual objects $X=\{x_{i}\}i=1,...,n$ with the corresponding values $y_{i}$ R of the target variable for object $X_{i}$.
    
- **Summary definition:** Answer is a real number.
    

---

### **Unsupervised Learning**

#### **Dimensionality Reduction**

- The goal of dimensionality reduction is to use some transformation functions to pass to the smallest number of new features based on the original features without losing any significant information about the sample objects.
    
- **Summary definition:** Answer is input data projection on low dimensional space.
    

#### **Clustering**

- The goal of clustering is to group objects into clusters using data on the pairwise similarity of objects.
    
- **Summary definition:** Answer is data division into groups.
    

---

### **Reinforcement Learning**

- The role of objects is played by the pairs <situation, decision>, the answers rs are the Learning Learning valuęsiafity functional characterizing the correctness of the decisions made (the reaction of the environment). _(Note: This exact wording reflects typographical artifacts present in the original document text)._

Here is a comprehensive study guide organized directly from the presentation lines. I have structured it to help you easily review the key concepts, formulas, and comparisons.


---

### 1. Linear Regression Basics

- **Definition:** Linear Regression is a supervised machine learning algorithm that is widely used to solve regression problems.
    
- **Goal:** The primary goal is to predict a continuous output variable based on one or more input variables. It does this by finding the best-fitting linear equation to describe the relationship between the inputs and the output.
    
- **Variables:**
    
    - **Independent Variable ($X$):** The input variable, also known as a feature. (e.g., Size of a house, Experience, Temperature) .
        
    - **Dependent Variable ($Y$):** The output variable we are trying to predict, also known as the target. (e.g., Price, Salary, Ice Cream Sales) .
        
- **The Best-Fitting Line:** The line that has the smallest difference between the predicted values and the actual values in the data points.
    

#### Types of Linear Regression

- **Simple Linear Regression:** Models the relationship using only one independent variable and one dependent variable.
    
    - **Equation:** $y=w_{0}+{w_{1}}^{*}x$.
        
    - **$w_{0}$:** The intercept term (the value of $y$ when $x$ is zero).
        
    - **$w_{1}$:** The slope coefficient (the change in $y$ for a unit change in $x$).
        
- **Multiple Linear Regression:** Models the relationship using multiple independent variables and one dependent variable.
    
    - **Equation:** $y=w_{0}+{w_{1}}^{*}X_{1}+...+{w_{n}}^{*}X_{n}$.
        
- **Polynomial Regression:** Used when data points form a curve rather than a straight line, modeling complex non-linear relationships.
    
    - **Concept:** The relationship between $x$ and $y$ is modeled as an n-th degree polynomial.
        
    - **Equation:** $y=w_{0}+w_{1}x+w_{2}x^{2}+\cdot\cdot\cdot+w_{d}x^{d}$.
        
    - **Example:** A movie's budget vs. box office revenue is non-linear (low budget = low revenue, medium budget = large increase, extremely high budget = returns grow slower due to market saturation).
        

---

### 2. The Loss Function

- **Purpose:** Also known as the cost function or objective function, it measures how well the model is performing by calculating the difference (residuals/errors) between predicted values and actual values.
    
- **Goal:** To minimize the loss function $L(w_{o},w_{1})\rightarrow min_{w0,w1}$.
    

#### Common Types of Loss Functions

1. **Mean Absolute Error (MAE):** $L=\frac{1}{n}\sum_{i=1}^{n}|f(x_{i})-y_{i}|$.
    
2. **Mean Square Error (MSE):** $L=\frac{1}{n}\sum_{i=1}^{n}(f(x_{i})-y_{i})^{2}$.
    

---

### 3. Finding the Best Parameters ($w_0$ and $w_1$)

To minimize the loss function, you could try waiting for a miracle, doing a grid search, or using mathematics.

#### The Analytical Solution (Normal Equation)

Instead of guessing, we can use linear algebra and calculus to find the exact minimum of the cost function.

- **Hypothesis:** $h_{\theta}(x^{(i)})=\theta_{0}x_{0}^{(i)}+\theta_{1}x_{1}^{(i)}+...+\theta_{j}x_{j}^{(i)}$.
    
- **Cost Function:** $J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}$.
    
- **Vectorized Cost Function:** $J(\theta)=(X\theta-y)^{T}(X\theta-y)$. (Note: the constant $1/(2m)$ is omitted as it doesn't affect the minimization result.)
    
- **The Derivative:** Taking the derivative of the expanded cost function with respect to $\theta$ yields: $\frac{\partial J(\theta)}{\partial\theta}=2X^{T}X\theta-2X^{T}y$.
    
- **The Final Solution:** Setting the derivative to zero and solving for $\theta$ gives the analytical solution: $\theta=(X^{T}X)^{-1}X^{T}y$.
    

**Why don't we always use the analytical solution?**

1. **Computational Complexity:** It requires calculating the inverse of the matrix $X^{T}X$, which has a time complexity of about $O(n^{3})$. This is too computationally expensive for large datasets.
    
2. **Distributed/Parallel Computing:** Matrix inversion is difficult to parallelize, making it inefficient in distributed environments.
    

---

### 4. Gradient Descent

Gradient Descent is an optimization algorithm used to iteratively adjust the intercept and slope coefficients ($w_0$ and $w_1$) to minimize the cost function.

- **Intuition:** Imagine trying to find the lowest point in a hilly terrain. You feel the slope under your feet and take small steps in the steepest downward direction until you reach the valley bottom.
    

#### Mathematical Requirements

The cost function must be:

1. **Differentiable:** The function must have a derivative for each point in its domain.
    
2. **Convex:** For a univariate function, the line segment connecting two points on the curve must lay on or above the curve without crossing it.
    

#### The Gradient

- The gradient tells us the direction of the steepest _increase_ and the rate of that increase.
    
- We move in the _opposite_ (negative) direction of the gradient because the slope points uphill.
    
- For univariate functions, it's the first derivative at a point.
    
- For multivariate functions, it's a vector of derivatives for each variable axis.
    ![[Pasted image 20260415193006.png|151]]

#### The Algorithm Steps

1. Initialize $w_{0}$ and $w_{1}$ to random values.
    
2. Calculate predicted values using current weights.
    
3. Calculate the cost function (predicted vs. actual).
    
4. Calculate the gradient of the cost function with respect to the weights.
    
5. Update the values of $w_{0}$ and $w_{1}$ by taking a step in the opposite direction of the gradient with a step size determined by the learning rate. Note: New values for all parameters must be calculated first, and then updated simultaneously.
    
    - **Formula:** $w^{k+1} = w^{k} - \alpha * \text{Gradient}$.
        
6. Repeat steps 2-5 until minimized or max iterations reached.

---

### 5. The Learning Rate ($\alpha$)

The learning rate ($\alpha$) determines the step size taken during gradient descent.

- **Too Low:** The algorithm will take tiny steps and be very slow to reach the minimum.
    
- **Too High:** The algorithm might overshoot the minimum and fail to converge.
    
- **Optimal:** Reaches the lowest loss efficiently.
    
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/learning_rate.html" width="100%" height="750" frameborder="0"></iframe>

#### How to pick a learning rate:

1. Trial and error method.
    
2. Choose commonly used values (e.g., $0.01, 0.001, 0.03, 0.003$).
    
3. Use learning rate schedulers.
    

#### Learning Rate Schedulers

- **StepLR:** Starts with an initial rate, then multiplies it by a decay factor after a specific number of training epochs (GD iterations) to reduce it.
    
- **CyclicLR:** Sets a minimum and maximum learning rate threshold, allowing the rate to fluctuate between the two in cycles. (Variants include Triangular, Triangular2, and Exp_range) .
    
- **ReduceLROnPlateau:** Drops the learning rate by a proportional decay factor only when a monitored metric stops improving (after a set "patience" period).
    
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/lr_schedulers.html" width="100%" height="750" frameborder="0"></iframe>

---

### 6. Types of Gradient Descent

There are three main variations based on how much data is used to calculate the gradient in a single step.

#### 1. Vanilla (Batch) Gradient Descent

Computes the gradient using the _entire_ training dataset for a single update.

- **Cons:** Can be very slow; intractable for datasets that do not fit into memory.
    

#### 2. Stochastic Gradient Descent (SGD)

Uses only a _single random training example_ $(x^{(i)}, y^{(i)})$ to calculate the gradient and update parameters per iteration.

- **Pros:** Computationally efficient; reduces memory footprint; able to escape local minima.
    
- **Cons:** High variance in updates; sensitive to hyperparameters; carries a risk of converging to suboptimal minima.
    

#### 3. Mini-batch Gradient Descent

Takes the best of both worlds by performing an update using a subset ("batch") of $n$ training examples per iteration.

- **Pros:** Computationally efficient; converges faster; reduces the variance of updates compared to SGD.
    
- **Cons:** The mini-batch size becomes a new hyperparameter; the model is sensitive to batch size; still carries a risk of convergence to suboptimal minima.

![[Pasted image 20260415201119.png]]


---

### 1. Classification Basics

In Machine Learning terminology, a classification model takes an **instance** as input and outputs a **label** (e.g., a Spam email detector takes an email as input and outputs a Positive or Negative label: spam / not spam).

- **Types of Classifiers:**
    
    - **Binary classifier:** Predicts between 2 labels ($n=2$).
        
    - **Multi-class classifier:** Predicts between multiple labels ($n \ge 2$).
        
- **Examples of Classification:**
    
    - Instance = photo $\rightarrow$ Labels = Fox / Dhole
        
    - Instance = audio $\rightarrow$ Labels = Speech / Music
        
    - Instance = text $\rightarrow$ Labels = Positive / Negative
        

#### Mathematical Notation

Notations can vary in the literature, but typically:

- **$X$**: Input space (set of all possible instances).
    
- **$Y$**: Output space (all possible labels).
    
- **$f:X \rightarrow Y$**: Any such function is a classifier.
    
- **$x \in X$**: Instance.
    
- **$y \in Y$**: Actual / true label of instance $x$.
    
- **$\hat{y} = f(x)$**: Predicted label of instance $x$.
    

**Evaluating the Classifier:**

We assume that every instance has an actual / true label $y$.

- A classifier is _perfect_ if it is always correct, predicting the actual label: $\hat{y}=y$.
    
- Usually, classifiers make some errors where for instance $i \in \{1,2,...,n\}$, the prediction does not match the reality: $\hat{y}_{i} \neq y_{i}$.
    

---

### 2. Building a Classifier (From Scratch to Linear)

#### A Non-ML Approach (Rule-Based)

- **What is a typical spam email?** It advertises unwanted things, tries to steal money, or tries to infect your computer. It often includes offers ("Lose weight fast"), impersonates legitimate institutions, claims you won a lottery, or has an odd sender address.
    
- **Our first spam detector:** Let's encode the presence of a feature with 1 and absence with 0.
    
    - Instance: $x=(0,0,1,0,...)$
        
    - Rule: Classify as spam whenever at least 5 features are present in the email.
        
    - Mathematical rule: $Classifier: \hat{y} = f(x) = \begin{cases} 1 \text{ (spam)} & \text{if } x_{1}+x_{2}+...+x_{n} \ge 5 \\ 0 \text{ (not spam)} & \text{otherwise} \end{cases}$
        

#### Adding Weights (Linear Classifier)

We can make some variables more important by adding weights ($w$). For example, if $x_{1}$ is three times more important and $x_{2}$ is two times more important:

$$f(x) = \begin{cases} 1 \text{ (spam)} & \text{if } w_{1}x_{1}+w_{2}x_{2}+...+w_{n}x_{n} \ge 5 \\ 0 \text{ (not spam)} & \text{otherwise} \end{cases}$$

_(Where $w_{1}=3$, $w_{2}=2$, and remaining weights $w_{i}=1$)_

- **Definition of a Linear Classifier:** A binary classifier is called a linear classifier if it predicts one class whenever $\sum w_{i}x_{i} \ge \text{threshold}$ and predicts the other class otherwise.
    
- **Finding the Weights Manually:** We want a classifier that makes as few mistakes as possible on a training dataset. Doing this manually is bad because it is difficult to scale with more features/emails, suffers from personalization problems, and requires too many resources.
    

#### Finding Weights Mathematically

We want to predict "spam" if an email has features similar to average spam emails, and "not spam" if it is similar to average non-spam emails.

Let $p$ be the center of positive instances, and $n$ be the center of negative instances.

- $p = \frac{1}{|Tr^{\oplus}|}\sum_{x \in Tr^{\oplus}}x$
    
- $n = \frac{1}{|Tr^{\ominus}|}\sum_{x \in Tr^{\ominus}}x$
    

We predict positive if instance $x$ is closer to $p$ than $n$. Using Euclidean distance:

$$||x-n|| \ge ||x-p||$$

$$\sqrt{(x-n)^{T}(x-n)} \ge \sqrt{(x-p)^{T}(x-p)}$$

After expanding the dot products and simplifying, we get:

$$(p-n) \cdot x \ge (p-n) \cdot (p+n)/2$$

This gives us our linear rule: **$w \cdot x \ge t$**

_(Where $w = p-n$ and $t = (||p||^{2}-||n||^{2})/2$)_

_Note: Even though a perfect linear classifier seems to exist, it is not always found by this basic linear classification method._

---

### 3. Logistic Regression

Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the _probability_ that an instance belongs to a given class or not.

- It can be viewed as a regression method restricted into a $[0, 1]$ interval.
    
- **The Problem:** We used Linear Regression when $y$ was continuous, but we need a way to transform the predictions so that we get binary labels (0 or 1) instead of continuous quantities.
    

#### The Sigmoid Function

To force the output ($z$) to be a legal probability between 0 and 1, we use the logistic sigmoid function:

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+\exp(-z)}$$

Where $z = w \cdot x + b$.

**Advantages of the Sigmoid Function:**

1. It outputs values between 0 and 1, making it ideal for binary classification where the output is interpreted as the probability of belonging to a certain class.
    
2. It is continuous and differentiable, which allows for the calculation of gradients necessary for optimization (Gradient Descent).
    
3. It naturally provides a decision boundary/threshold (usually 0.5).
    
    - $decision(x) = \begin{cases} 1 & \text{if } P(y=1|x) > 0.5 \\ 0 & \text{otherwise} \end{cases}$
        

#### Example Calculation (Sentiment Classification)

Suppose we want to classify a movie review as Positive (1) or Negative (0).

- Features ($x$): Word counts, pronoun counts, etc. Let $x = [3, 2, 1, 3, 0, 4.19]$
    
- Weights ($w$): Learned importance of features. Let $w = [2.5, -5.0, -1.2, 0.5, 2.0, 0.7]$
    
- Bias ($b$): $0.1$
    
- **Probability it is Positive:** $P(+|x) = \sigma(w \cdot x + b) = \sigma(0.833) = 0.70$
    
- **Probability it is Negative:** $P(-|x) = 1 - \sigma(w \cdot x + b) = 0.30$
    

---

### 4. Loss Function and Optimization

We want to learn parameters ($w$ and $b$) that make the prediction ($\hat{y}$) for each training observation as close as possible to the true label ($y$).

#### Why don't we use Mean Squared Error (MSE)?

1. MSE assumes that data was sampled from a Gaussian distribution, but data for binary classification is assumed to be from two categories (Bernoulli distribution).
    
2. MSE leads to non-convex functions in classification, making it hard for gradient descent to find the global minimum.
    
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/loss_functions.html" width="100%" height="750" frameborder="0"></iframe>

#### The Log-Likelihood Loss Function (Cross-Entropy)

We want to maximize the probability of the correct label. Since outcomes are 0 or 1 (Bernoulli), we express the probability as:

$$p(y|x) = \hat{y}^{y}(1-\hat{y})^{1-y}$$

To make the math easier, we take the log of both sides (values that maximize a probability also maximize its log):

$$\log p(y|x) = y \log\hat{y} + (1-y)\log(1-\hat{y})$$

This describes a log likelihood that should be _maximized_. To turn it into a loss function ($L$) that we want to _minimize_, we flip the sign:

$$L(\hat{y},y) = -[y \log\hat{y} + (1-y)\log(1-\hat{y})]$$

---

### 5. Derivatives for Gradient Descent

To iteratively update the weights and minimize the loss function, we use Gradient Descent, which requires finding the derivative of the loss function with respect to the weights ($\frac{dL}{dw}$).

**1. Derivative of the Sigmoid Function:**

Given $\sigma(z) = \frac{1}{1+e^{-z}}$ and $1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}$, the derivative is:

$$\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1-\sigma(z))$$

**2. Derivative of the Loss Function (Chain Rule):**

We apply the chain rule to find $\frac{dL}{dw}$:

$$\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz} \cdot \frac{dz}{dw}$$

Breaking it down into parts:

- $\frac{dL}{d\hat{y}} = -(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}) = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$
    
- $\frac{d\hat{y}}{dz} = \hat{y}(1-\hat{y})$ _(From the sigmoid derivative above)_
    
- $\frac{dz}{dw} = x$ _(Because $z = wx+b$)_
    

**Final Gradient Formula:**

Multiplying the parts together simplifies beautifully to:

$$\frac{dL}{dw} = (\hat{y}-y) \cdot x$$

---

### 1. Multiclass Classification Overview

Unlike binary classification (which separates two classes), **Multiclass Classification** assigns each input to exactly _one_ class out of three or more possible categories.

- **Mathematical Definition:** Given an input feature vector $X \in R^{n}$ and a target variable $y \in \{1,2,...,k\}$, the goal is to learn a function $f:X \rightarrow \{1,2,...,k\}$ that maps inputs to one of the $k$ classes.
    
- **Examples:** Handwritten digit recognition, species classification, sentiment analysis, disease diagnosis.
    

---

### 2. Multinomial Logistic Regression (MLR)

MLR is a generalization of binary logistic regression used for tasks with more than two categories. It models the probability of each class as a function of the input features.

#### Why can't we just use the Sigmoid function?

- **Independent Treatment:** Sigmoid treats each class independently.
    
- **No Probability Constraint:** If we use sigmoid for $K$ classes, there is no constraint that the probabilities sum to 1 ($\sum_{k=1}^{K}P_{k}=1$).
    
- **No Competition:** In multiclass classification, if the probability of one class increases, the probabilities of the other classes _must_ decrease. Sigmoid does not enforce this competition.
    

#### The Solution: The Softmax Function

Given a dataset with $K$ classes, MLR uses the **Softmax function** to model probabilities.

- **Formula:** $P(y=k) = \frac{e^{z_{k}}}{\sum_{j=1}^{K} e^{z_{j}}}$
    
- **Properties:**
    
    - Outputs valid probabilities between 0 and 1.
        
    - The sum of all probabilities across all classes is exactly 1.
        
    - Compares all raw logits to each other, unlike sigmoid.
        

Steps to Compute MLR

1. **Initialize the Weight Matrix ($W$):** For $n$ features and $K$ classes, the matrix size is $(n+1) \times K$. Each column represents the weights for a specific class.
    
2. **Calculate Linear Scores (Logits):** Calculate the raw score $z$ for every class $k$: $z_{k} = w_{k}^{T}x + b_{k}$.
    
3. **Apply Softmax:** Convert raw scores into probabilities that sum to 1.
    

#### The Loss Function: Cross-Entropy Loss

- The model is trained by minimizing the negative log-likelihood, also known as Cross-Entropy Loss (or Softmax Loss).
    
- It measures the difference between the true class labels and the predicted probability distribution.
    
- **Formula:** $J = -\frac{1}{N}(\sum_{i=1}^{N}y_{i} \cdot \log(\hat{y}_{i}))$.
    

---

### 3. Multiclass Strategies

When dealing with algorithms inherently designed for binary classification, we can adapt them for multiclass problems using two main strategies.

#### Strategy A: One vs One (OvO)
![[Pasted image 20260416175308.png]]

Trains a separate binary classifier for **every possible pair** of classes.

- **Number of Classifiers:** If there are $N$ classes, you need $N*(N-1)/2$ classifiers.
    
- **Prediction:** Uses a **majority voting mechanism**. The class that wins the most votes across all binary classifiers is the final prediction.
    
- **Pros:** Each classifier works with a smaller dataset, making it computationally efficient for some models.
    
- **Cons:** Expensive to train and store for a large $N$; increases inference time because you have to run so many models.
    

#### Strategy B: One vs All / One vs Rest (OvA / OvR)
![[Pasted image 20260416175419.png]]
Trains a separate binary classifier for **each class against all other classes combined**.

- **Number of Classifiers:** Exactly $N$ classifiers for $N$ distinct classes.
    
- **Prediction:** The classifier that outputs the highest confidence or probability determines the predicted class.
    
- **Pros:** Requires only $K$ classifiers (computationally cheaper and simpler to implement than OvO).
    
- **Cons:** Suffers from **class imbalance** because each model trains on one class vs a large, consolidated negative class; decision boundaries may be less accurate.
    

---

### 4. Multilabel Classification

Unlike multiclass classification (where an instance belongs to exactly one class), **Multilabel Classification** allows each instance to be assigned **multiple labels simultaneously**.

- **Mathematical Definition:** Maps each input to a subset of labels: $f:X \rightarrow 2^{y}$ where $y=\{y_{1}, y_{2}, ..., y_{k}\}$.
    
- **Mechanism:** We treat the prediction of $K$ labels as $K$ separate binary classification tasks happening at the same time.
    
- **Activation Function:** We do **not** use Softmax because the labels are not mutually exclusive. Instead, we use the **Sigmoid** function for each score individually to get independent probabilities: $\hat{y}_{k} = \sigma(z_{k}) = \frac{1}{1+e^{-z_{k}}}$.
    
- **Loss Function:** We use **Binary Cross-Entropy (BCE)** because of the fundamental assumption of label independence.
![[Pasted image 20260416175708.png]]

![[Pasted image 20260416175505.png]]

| **Feature**              | **Sigmoid Function Classifications**                                                                                                     | **Softmax Function Classifications**                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Classification Type**  | **Binary Classification** (predicting between 2 distinct classes)<br>**Multilabel Classification**                                       | **Multiclass Classification** (predicting between 3 or more distinct, mutually exclusive classes) |
| **Mutual Exclusivity**   | **No** (predictions for each class are independent) - an input could theoretically score high for multiple independent binary attributes | **Yes** (an input can only belong to **one** class out of the many possible options)              |
| **Sum of Probabilities** | **Do not necessarily sum to 1** across different independent classes                                                                     | **Always sum to 1 exactly** (Softmax normalizes the probabilities so they add up to 100%)         |
| **Primary Use Case**     | Determining a single yes/no, present/absent, or positive/negative label for _each_ category separately                                   | Categorizing an input into one specific bucket among several non-overlapping possibilities        |
| **Common Example**       | Independently predicting if an email is **Spam** (yes/no) and if it is **Urgent** (yes/no)                                               | Classifying an image as either a **Dog**, **Cat**, or **Horse** (it must be exactly one of them)  |

Think of Sigmoid as a series of independent questions being asked ("Is it spam? Is it urgent?"), while Softmax is a single competitive decision ("Which one animal is this?").

---

## 5: Data & Features


PLAN

1. Data
    
2. Feature Types
    
3. Feature Encoding
    
4. Feature Scaling
    
5. Text Representation
    
6. Image Representation
    

---

1. Data

**Definition:**

> Data the quantities, characters, or symbols on which operations are performed by a computer, which may be stored and transmitted in the form of electrical signals and recorded on magnetic, optical, or mechanical recording media.

The Machine Learning Pipeline

**Data Preparation** ➔ **Feature Extraction** ➔ **Model Training** (with a **Verification** loop back to Feature Extraction/Data Prep) ➔ **Predictions** (with a loop back to Data Prep).

Data Types to Numerical Representation

To be processed, all data formats must be converted:

- **Text** ➔ Numerical Representation
    
- **Audio** ➔ Numerical Representation
    
- **Statistical data** ➔ Numerical Representation
    
- **Video** ➔ Numerical Representation
    
- **Image** ➔ Numerical Representation
    

The Golden Rule of Data

- Garbage in, garbage out - if your data is flawed, your model's results will be too.
    
- _(DATA ISSUES EVERYWHERE)_
    
- High-quality data is essential for training effective deep learning models.
    
- Data should be accurate, consistent, and representative of the problem you're trying to solve.
    
- More data allows the model to learn complex patterns and variations in the data.
    

Data Preparation Steps

_(MORE DATA, MORE PROBLEMS)_

- Data cleaning
    
- Data conversion
    
- Feature engineering & selection
    
- Data normalization and standardization
    

Data Cleaning

- Detect and remove duplicate records
    
- Outlier detection
    
- Missing data handling
    

Missing Data Handling Methods

1. Deletion:

- **Record Deletion:** Remove entire records where any single value is missing. (This is easy but can lead to significant data loss.)
    
- **Column Deletion:** Delete the feature column if it contains a lot of missing values and feature is not crucial for given task.
    

2. Imputation:

- **Mean/Median/Mode Imputation:** Replace missing values with the mean, median, or mode of the column. (numerical data)
    
- **Predictive Modeling:** Use machine learning algorithms to predict and fill in missing values based on other data points.
    

---

2. Feature Types

**FEATURES**

1. **Numeric**
    
    - **Discrete:** (age, 24)
        
    - **Continues:** (height, 78.6)
        
2. **Categorical**
    
    - **Nominal:** (city, horse)
        
    - **Ordinal:** (good/bad)
        

**Examples of Feature Types Table:**

|**Nominal**|**Nominal**|**Discrete**|**Continues**|**Ordinal**|
|---|---|---|---|---|
|**Full Name**|**Sex**|**Age**|**Income**|**Building Type**|
|John Smith|Male|35|$50.000|Finished|
|Emily Johnson|Female|18|$15.870|Initial stage|
|Sarah Williams|Female|27|$76.940|Initial stage|
|Michael Davis|Male|42|$25.000|Finished|
|Robert Green|Male|53|$80.300|Under construction|

---

3. Feature Encoding

- Many machine learning algorithms cannot operate on label data directly.
    
- They require all input variables and output variables to be numeric.
    
- In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves.
    
- This means that categorical data must be converted to a numerical form.
    

Method A: Integer Encoding

Each unique category value is assigned an integer value. _(Example: Male ➔ 1, Female ➔ 2. Finished ➔ 3, Initial stage ➔ 1, Under construction ➔ 2)_

|**PROS**|**CONS**|
|---|---|
|**Simplicity:** Integer encoding is straightforward to implement, where each unique category value is simply assigned an integer.|**Implicit Ordering:** It introduces an arbitrary ordinal relationship among the categories, which may not be present in the data.|
|**Efficiency:** It results in a more compact representation, requiring less memory and computational power.|**Limited Model Applicability:** Many machine learning models may interpret the numerical values as having a mathematical relationship, which can affect the model's accuracy negatively.|

Method B: One-Hot Encoding

- For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.
    
- Using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results.
    
- In this case, a one-hot encoding can be applied to the integer representation.
    
- One-Hot (Unitary) encoding replaces a categorical feature with N values with N discrete numeric features that take the value 0 or 1 depending on the value of the original feature.
    

_(Example transformations)_

- Male ➔ `Sex_1 = 1, Sex_2 = 0`
    
- Female ➔ `Sex_1 = 0, Sex_2 = 1`
    
- Finished ➔ `Type_1 = 0, Type_2 = 0, Type_3 = 1`
    
- Initial stage ➔ `Type_1 = 1, Type_2 = 0, Type_3 = 0`
    

|**PROS**|**CONS**|
|---|---|
|**Eliminates Ordinal Relationships:** It prevents the model from assuming an unintended order among categories.|**Dimensionality Increase:** It can significantly increase the dataset's dimensionality, especially with categorical variables that have a large number of levels.|
|**Model Compatibility:** It is widely compatible with various types of machine learning models, especially those that cannot inherently handle categorical data.|**Sparsity:** The resulting data is often sparse, meaning that there are many zeros. This can be inefficient in terms of storage and computation.|

Feature Combinations

- Using the sum, difference, or product of two or more features can improve model results.
    
    - $y = x_{1} * x_{2}$
        
    - $x_{3} = x_{1} - x_{1}x_{2}$
        
    - $x_{4} = x_{2} - x_{1}x_{2}$
        
- Raising a feature to power and using it as a new feature can also improve model results.
    
    - $x_{2} = x_{1}^{2}$
        

---

4. Feature Scaling

Feature scaling is a data preprocessing technique used to transform the values of features or variables in a dataset to a similar scale.

- The purpose is to ensure that all features contribute equally to the model and to avoid the domination of features with larger values.
    
- Feature scaling becomes necessary when dealing with datasets containing features that have different ranges, units of measurement, or orders of magnitude.
    
- In such cases, the variation in feature values can lead to biased model performance or difficulties during the learning process.
    
- Gradient descent converges faster with feature scaling than without it.
    
- Some machine learning algorithms will not work properly without scaling.
    

Normalization vs Standardization

1. Normalization (Min-Max scaling)

- Normalization (Min-Max scaling) is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1.
    
- **Formula:** $X' = \frac{X - X_{min}}{X_{max} - X_{min}}$
    
- _Example Calculation:_ $35 \rightarrow (35-18)/(53-18) = 0.32$
    

2. Standardization

- Standardization is another feature scaling method where the values are centered around the mean with a unit standard deviation.
    
- This means that the mean of the attribute becomes zero, and the resultant distribution has a unit standard deviation.
    
- **Formula:** $X' = \frac{X - \mu}{\sigma}$
    
- _Example Calculation:_ $35 \rightarrow (35-35)/12.05 = 0$
    

|**Normalization**|**Standardization**|
|---|---|
|Rescales values to a range between 0 and 1|Centers data around the mean and scales to a standard deviation of 1|
|Useful when the distribution of the data is unknown or not Gaussian|Useful when the distribution of the data is Gaussian or unknown|
|Sensitive to outliers|Less sensitive to outliers|
|Retains the shape of the original distribution|Changes the shape of the original distribution|
|May not preserve the relationships between the data points|Preserves the relationships between the data points|

Here is an interactive calculator so you can see exactly how the original age data ($18, 27, 35, 42, 53$) transforms under both methods side-by-side:

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/feature_scaling.html" width="100%" height="750" frameborder="0"></iframe>


---

5. Text Representation

Bag of Words

- The Bag of Words method treats a document (which could be a sentence, paragraph, or an entire text) as an unordered collection or "bag" of words, disregarding grammar, word order, and context.
    
- It represents each document as a numerical vector, with each dimension corresponding to a unique word in the entire corpus of documents.
    
    - _Example:_ "Today is a great day. Today is a sunny day." ➔ `Today: 2, great: 1, day: 2, Sunny: 1`
        

**Steps:**

1. **Tokenization:** The first step is to split the text into individual words or tokens. Tokenization can be done using whitespace, punctuation, or more advanced techniques like word embeddings.
    
2. **Vocabulary Creation:** Build a vocabulary, which is a list of all unique words (or tokens) in the entire corpus of documents. Each word in the vocabulary is assigned a unique index.
    
3. **Vectorization:**
    
    - For each document, create a numerical vector of fixed length, typically equal to the size of the vocabulary.
        
    - Initialize all vector elements to zero.
        
    - For each word in the document, increment the corresponding element in the vector.
        
    - The resulting vector represents the frequency of each word in the document.
        

_(Example Vectors)_ `["I", "love", "natural", "language", "processing", "text", "analysis", "is", "interesting", "enjoy", "studying", "NLP"]`

- "I love natural language processing." ➔ `[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]`
    
- "Text analysis is interesting." ➔ `[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]`
    
- "I enjoy studying NLP." ➔ `[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]`
    

|**PROS**|**CONS**|
|---|---|
|**Simplicity:** The BoW model is simple to implement and easy to understand.|**Loss of Word Order:** The BoW disregards the order of words in a document.|
|**Flexibility:** The BoW model can be applied to a wide range of NLP tasks, including sentiment analysis, topic modelling, and text classification.|**Sparsity:** The BoW representation results in high-dimensional and sparse feature vectors.|
|**Efficiency:** BoW representations can be quickly computed for a large number of documents.|**Semantic Information:** BoW doesn't capture semantic relationships between words. It treats each word as an independent feature.|
|**Interpretability:** The BoW model provides a clear and interpretable representation of text data, where each word is treated as a separate feature.|**Out-of-Vocabulary Words:** Words that are not in the vocabulary are typically ignored in the BoW representation.|

Text Processing

Text preprocessing usually includes the following steps before feature extraction:

- Splitting text into sentences
    
- Splitting sentences into words
    
- Remove punctuation (optional)
    
- Remove stop words (optional)
    
- Convert words to lowercase (optional)
    
- Word normalization (stemming, lemmatization) (optional)
    

**Stop Words** Stop words are words that are commonly used in a language but are often removed from text data during pre-processing because they are considered to be of little value in terms of understanding the meaning or context of a text. _(e.g., "a", "an", "the", "in", "on", "of", "for", "to", "and", etc.)_

**Normalization**

- **Stemming** is the process of reducing a word to its base or root form.
    
- **Lemmatization** is the process of reducing a word to its base or dictionary form.
    

|**Word**|**Stemma**|**Lemma**|
|---|---|---|
|running|run|run|
|leaves|leav|leave|
|fishing|fish|fish|
|brighter|brighter|bright|
|cars|car|car|
|played|play|play|
|better|better|good|
|children|children|child|
||||

---

6. Image Representation

- **Grayscale:** Represented as a 2D array of pixel intensities (0-255).
    
- **Color:** Represented as 3D arrays (R, G, B channels).
    

Data Augmentation

Deep learning models typically require large amounts of data to generalize well. More data allows the model to learn complex patterns and variations in the data. Data augmentation techniques can be applied to artificially increase the size of your training dataset. This involves creating new training examples by applying various transformations to the existing data.

**Text Augmentation:**

- **Synonyms and Antonyms:** Replace words with synonyms or antonyms to create variations in text.
    
- **Random Insertion:** Insert random words or phrases into sentences or paragraphs.
    
- **Random Deletion:** Remove random words from sentences or paragraphs.
    
- **Shuffling:** Shuffle the order of words or sentences in the text.
    
- **Masking:** Mask certain words or phrases with placeholders.
    
- **Back Translation:** Translate text to another language and then back to the original language.
    

**Image Augmentation:**

- **Rotation:** Rotate images by various angles to create new versions of the same image.
    
- **Flipping:** Horizontally flip images to generate mirror images.
    
- **Zooming:** Apply zoom operations to crop and resize images.
    
- **Scaling:** Resize images to different scales while maintaining their aspect ratio.
    
- **Brightness and Contrast:** Adjust brightness and contrast levels in images.
    
- **Color Variations:** Apply color distortions or changes, such as changing color balance or saturation.
    
- **Noise:** Add various types of noise, like Gaussian or salt-and-pepper noise, to images.
    
- **Cropping:** Crop different regions of an image to focus on specific features.
    

_(HOW TO CONFUSE MACHINE LEARNING: Images of dogs vs fried chicken, chihuahuas vs blueberry muffins)_

---
## Metrics & Over/Underfitting

PLAN

|**Section**|**Topic**|
|---|---|
|**01**|Data Splitting|
|**02**|Cross-Validation|
|**03**|Metrics|
|**04**|Overfitting|
|**05**|Underfitting|
|**06**|Bias and Variance|
|**07**|Regularization|

---

### 01. Data Splitting

### The Dataset Breakdown (Visual Diagram)

A dataset is typically broken down into subsets to train and test the model:

- **Dataset** ➔ **Train set** + **Test set**
    
- **Dataset** ➔ **Train set** + **Validation set** + **Test set**
    

### The Machine Learning Loop (Flowchart)

1. **Train model on Training Set** ➔
    
2. **Evaluate model on Validation Set** ➔
    
3. **Tweak model according to results on Validation Set** ➔ _(Loops back to step 1)_
    
4. **Pick model that does best on Validation Set** ➔
    
5. **Confirm results on Test Set**
    

How to split data?

The truth is - There is no optimal split percentage.

**Common Split Ratios (Visual Breakdown):**

|**Training data**|**Validation data**|**Test data**|
|---|---|---|
|80%|10%|10%|
|70%|15%|15%|
|60%|20%|20%|

Data Splitting Techniques

|**Random**|**Stratified**|
|---|---|
|The dataset is shuffled, and samples are picked randomly and put in the train, validation, or the test set based on what percentage ratio is given by the user.|Stratified random sampling is done by dividing the data into homogeneous groups called strata. Stratified sampling involves taking random samples from stratified groups, in proportion to the data.|
|For example, if a dataset has 1000 images, of which 800 "dog" and 200 "cat," and the data split into training and test sets in an 80%-20% ratio, it might so happen that the training set consists only dog images, while the test set consists of only cat images.|For example, a dataset consists of 1000 images, of which 600 dog and 400 cat images. Stratified sampling ensures that 60% of the images are of category "dog" and 40% are of category "cat" in the training and validation sets.|

---

### 02. Cross-Validation

- Whenever we build any machine learning model, we feed it with initial data to train the model.
    
- And then we feed some unknown data (test data) to understand how well the model performs and generalized over unseen data.
    
- If the model performs well on the unseen data, it's consistent and is able to predict with good accuracy on a wide range of input data; then this model is stable.
    
- But this is not the case always! Machine learning models are not always stable and we have to evaluate the stability of the machine learning model.
    
- That is where Cross-Validation comes into the picture.
    

K-fold Cross-Validation

K-fold cross-validation is one of the most common techniques. It allows to train and test your model k-times on different subsets of training data and build up an estimate of the performance of a model on unseen data.

**Visual Diagram of K-fold:**

![[Pasted image 20260416183942.png]]
    

Stratified K-fold

This is a slight variation from K-Fold Cross-Validation, which uses 'stratified sampling' instead of 'random sampling.' _(Class Distributions M and F remain perfectly balanced across Round 1, Round 2, Round 3, Round 4, and Round 5)._
![[Pasted image 20260416184153.png]]
Cross-Validation: PROS vs CONS

|**PROS**|**CONS**|
|---|---|
|Provides a more accurate and reliable measure of model performance by averaging results across multiple subsets.|Training and evaluating the model multiple times increases computational time and resource usage.|
|Helps identify if a model generalizes well or if it fits too closely to training data.|With large datasets, computational costs may outweigh the incremental benefits.|
|Makes efficient use of limited data.||
|Helps select the best parameters by comparing different model configurations.||

---

### 03. Metrics

1. Accuracy

Accuracy measures the percentage of correctly predicted instances out of the total instances in a dataset.

- **Formula:** $Accuracy = (1/n)\Sigma[y_{pred}=y]$
    
- **PROS:** Easy to interpret.
    
- **CONS:** Accuracy does not work for imbalance datasets.
    

**Accuracy Imbalance Example:**

- Classifier should classify whether the student will pass the course?
    
- There are 880 students who won't pass and 120 who will pass in the test set.
    
- What if a trivial classifier predicts that all students won't pass?
    
- $Accuracy = 880/1000 = 0.88$
    

2. Confusion Matrix
![[Pasted image 20260416184338.png]]
3. Precision

Precision measures the accuracy of positive predictions made by a model, specifically the ratio of correctly predicted positive instances to the total number of instances predicted as positive.

- Precision is especially useful when you want to assess how well a model performs in minimizing false positives.
    
- **Formula:** $Precision=TP/(TP+FP)$
        

4. Recall (Sensitivity, TPR)

Recall (Sensitivity, TPR) measures the ability of a model to correctly identify all true positives out of all the actual positive instances in a dataset.

- Recall is particularly useful when you want to assess how well a model captures and doesn't miss positive instances.
    
- **Formula:** $Recall = TP / (TP + FN)$
    

5. Specificity

Specificity (TNR) measures the ability of a model to correctly identify negative instances.

- It is the complement of the false positive rate and is often used in conjunction with sensitivity (recall) to assess a model's performance.
    
- **Formula:** $Specificity=TN/(TN+FP)$
![[Pasted image 20260416205348.png|361]]

6. F1-score

F1-score combines precision and recall into a single metric, providing a balanced assessment of a model's performance.

- The F1 score is especially useful when you want to strike a balance between precision and recall and when the class distribution is imbalanced.
    
- **Formula:** $F1-score=2*precision*recall/(precision+recall)$
    

7. ROC-AUC

The Receiver Operating Characteristic-Area Under the Curve (ROC-AUC) is an evaluation metric commonly used for assessing the performance of binary classification models, particularly in scenarios where you want to understand the model's ability to discriminate between positive and negative classes across different thresholds.

- The ROC curve is created by plotting the TPR (sensitivity) on the y-axis against the FPR on the x-axis as the classification threshold varies.
    
- **Formulas:** * $TPR=\frac{TP}{TP+FN}$
    
    - $FPR=\frac{FP}{FP+TN}$
        
- **Interpretation:**
    ![[Pasted image 20260416205649.png]]

    

Metrics: Averaging

**Averaging method:**

- **Macro Averaging:** Unweighted average of metrics for each class.
    
- **Micro Averaging:** Calculated globally by counting total TP, FP, FN.
    

**Averaging Example Table: True Labels vs Predicted Labels**

![[Pasted image 20260416212528.png]]
![[Pasted image 20260416212904.png]]
    

**Metrics Questions:**
1. Hospital wants to avoid alarming healthy patients, which metric is more suitable?

**Answer:** **Precision** (and/or **Specificity**).

**Why?** * "Alarming a healthy patient" means the model predicted they were sick (Positive) when they were actually healthy (Negative). In machine learning terms, this is a **False Positive (FP)**.

- The hospital's goal is to minimize False Positives.
    
- **Precision** measures how accurate your positive predictions are ($TP / (TP + FP)$). If you have high precision, it means that when the model says "You are sick," it is almost certainly correct, and healthy people won't get scary, false-alarm phone calls.
    
- _(Note: **Specificity** is also a great answer here, as it directly measures the model's ability to correctly identify the true negatives—the healthy patients)._
    

If the hospital had the opposite goal—wanting to make absolutely sure they don't miss a single sick person, even if it means accidentally alarming a few healthy people—they would optimize for **Recall (Sensitivity)** to minimize False Negatives.

 2. If the disease occurs in only 5% of the population, which metric is not suitable?

**Answer:** **Accuracy**.

**Why?**

- A disease occurring in only 5% of the population means you have a highly **imbalanced dataset**.
    
- As noted in your slides, accuracy is not an effective metric for imbalanced data.
    
- Think about the student pass/fail example from your lecture: a "trivial classifier" could just predict that _nobody_ passes and still get 88% accuracy.
    
- In this hospital scenario, a terrible, lazy model could simply predict "Healthy" for 100% of the patients. Because 95% of the population is naturally healthy, this useless model would achieve an incredible **95% Accuracy**, despite failing to catch a single sick patient!

    
![[Pasted image 20260416214801.png]]
    

**Model A Metrics:**

- **Recall (Sensitivity):** TP / (TP + FN) = 45 / (45 + 5) = **90%**
    
- **Precision:** TP / (TP + FP) = 45 / (45 + 95) = **~32%**
    
- **Specificity:** TN / (TN + FP) = 855 / (855 + 95) = **90%**
    

**Model B Metrics:**

- **Recall (Sensitivity):** TP / (TP + FN) = 35 / (35 + 15) = **70%**
    
- **Precision:** TP / (TP + FP) = 35 / (35 + 20) = **~64%**
    
- **Specificity:** TN / (TN + FP) = 930 / (930 + 20) = **~98%**
    

1. Which model would you choose for medical screening?

**Answer: Model A**

**Why?** The primary goal of a preliminary medical _screening_ (like a general blood test or airport security scanner) is to cast a wide net and make sure you **do not miss anyone who is actually sick**. You want to minimize False Negatives.

- **Model A** has a much higher **Recall (90%)** compared to Model B (70%). It successfully catches 45 out of the 50 sick patients, whereas Model B completely misses 15 of them.
    
- It is acceptable that Model A has terrible Precision (32%) and flags 95 healthy people as sick (False Positives), because those people will just be sent for a second, more accurate test anyway.
    

2. Which model would you choose for confirmatory diagnosis?

**Answer: Model B**

**Why?** A _confirmatory diagnosis_ happens after a screening. It is usually an expensive, invasive, or dangerous procedure (like a biopsy, strong medication, or surgery). At this stage, you need to be absolutely certain the patient is sick before proceeding. You want to minimize False Positives.

- **Model B** has much higher **Precision (~64%)** and **Specificity (~98%)**.
    
- When Model B says a patient is positive, it is much more likely to be correct. It only falsely alarms 20 healthy people, compared to Model A which would have sent 95 healthy people to unnecessary surgery!
    

---

### 04. Overfitting & 05. Underfitting

![[Pasted image 20260416215029.png]]

Overfitting

Overfitting is a common pitfall in deep learning algorithms, in which a model tries to fit the training data entirely and ends up memorizing the data patterns and the noise/random fluctuations. These models fail to generalize and perform well in the case of unseen data scenarios, defeating the model's purpose.


**Overfitting: Causes**

- The training data is not cleaned and contains some "garbage" values. The model captures the noise in the training data and fails to generalize the model's learning.
    
- The training data size is insufficient, and the model trains on the limited training data for several epochs.
    
- Model is complex and require a significant amount of time to train, and often lead to overfitting the training set.
    
- Incorrect tuning of hyperparameters in the training phase leads to over-observing the training set, resulting in memorizing features.
    

**Overfitting: Prevention**

1. **Adding more data:** Most of the time, adding more data can help deep learning models detect the "true" pattern of the model, generalize better, and prevent overfitting.
    
2. **Early stopping:** In iterative algorithms, it is possible to measure how the model iteration performance. Up until a certain number of iterations, new iterations improve the model. After that point, however, the model's ability to generalize can deteriorate as it begins to overfit the training data. Early stopping refers to stopping the training process before the learner passes that point.
    
3. **Data augmentation:** Data augmentation techniques increase the amount of data by slightly changing previously existing data and adding new data points or by producing synthetic data from a previously existing dataset.
    
4. **Remove features:** Irrelevant aspects can be deleted from data to improve the model. Many characteristics in a dataset may not contribute much to prediction. Removing non-essential characteristics can enhance accuracy and decrease overfitting.
    
5. **Ensembling:** It is a machine learning technique that combines several base models to produce one optimal predictive model. In Ensemble Learning, the predictions are aggregated to identify the most popular result.
    

Underfitting

Underfitting is another common pitfall in machine learning, where the model cannot create a mapping between the input and the target variable. Under-observing the features leads to a higher error in the training and unseen data samples.

**Underfitting: Causes**

- Unclean training data containing noise or outliers can be a reason for the model not being able to derive patterns from the dataset.
    
- The model is assumed to be too simple.
    
- Incorrect hyperparameters tuning often leads to underfitting due to under-observing of the features.
    

**Underfitting: Detection & Prevention**

1. **Training and test loss:** If the model is underfitting, the loss for both training and validation will be considerably high. In other words, for an underfitting dataset, the training and the validation error will be high.
    
2. **Oversimplistic prediction graph:** If a graph with the data points and the fitted curve is plotted, and the classifier curve is oversimplistic, than, most probably, your model is underfitting. In those cases, a more complex model should be tried out.
    
3. **Train a more complex model:** Lack of model complexity in terms of data characteristics is the main reason behind underfitting models.
    
4. **More time for training:** Early training termination may cause underfitting. The number of epochs should be increased to get better results.
    
5. **Eliminate noise from data:** Another cause of underfitting is the existence of outliers and incorrect values in the dataset. Data cleaning techniques can help deal with this problem.
    
6. **Try a different model:** If none of the above-mentioned principles work, a different model should be different (usually, the new model must be more complex by its nature).
    

---

### 06. Bias and Variance

Bias

Bias:

1. measures the difference between the model's prediction and the target value.
    
2. refers to prediction error that is introduced in the model due to oversimplifying the machine learning algorithms.
    

If the model is oversimplified, the predicted value would be far from the ground truth resulting in more bias.

**Characteristics of a high-bias model include:**

- Failure to capture proper data trends
    
- Potential towards underfitting
    
- More generalized/overly simplified
    
- High error rate
    

Variance

Variance:

- is the error due to the model's sensitivity to fluctuations in the training data
    
- means that model performs well with the training dataset, but does not perform well with the test dataset.
    
- High variance occurs when a model learns the training data's noise and random fluctuations rather than the underlying pattern.
    

**A model with high variance has the below problems:**

- A high variance model leads to overfitting.
    
- Increased model complexities.
    

The 4 Scenarios (Visual Target Metaphor)
![[Pasted image 20260416220819.png|260]]
1. **Low-Bias, Low-Variance:** The combination of low bias and low variance shows an ideal machine learning model. However, it is not possible practically. _(Visual: All points tightly clustered in the center bullseye)_
    
2. **Low-Bias, High-Variance:** With low bias and high variance, model predictions are inconsistent and accurate on average. This case occurs when the model learns with a large number of parameters and hence leads to an overfitting. _(Visual: Points widely scattered but generally centered around the bullseye)_
    
3. **High-Bias, Low-Variance:** With High bias and low variance, predictions are consistent but inaccurate on average. This case occurs when a model does not learn well with the training dataset or uses few numbers of the parameter. It leads to underfitting problems in the model. _(Visual: Points tightly clustered together, but completely missing the bullseye off to the side)_
    
4. **High-Bias, High-Variance:** With high bias and high variance, predictions are inconsistent and also inaccurate on average. _(Visual: Points widely scattered and completely missing the bullseye off to the side)_
    

Bias-Variance Trade-Off

- For an accurate prediction of the model, algorithms need a low variance and low bias.
    
- But this is not possible because bias and variance are related to each other:
    
    - If we decrease the variance, it will increase the bias.
        
    - If we decrease the bias, it will increase the variance.

    

**The Mathematical Trade-off:**

$MSE = (Y-\hat{Y})^2$ $= (Y-E(\hat{Y})+E(\hat{Y})-\hat{Y})^2$ $= (Y-E(\hat{Y}))^2+(E(\hat{Y})-\hat{Y})^2+2(Y-E(\hat{Y}))(E(\hat{Y})-\hat{Y})$ $E[(Y-\hat{Y})^2] = E[(Y-E(\hat{Y}))^2+(E(\hat{Y})-\hat{Y})^2+2(Y-E(\hat{Y}))(E(\hat{Y})-\hat{Y})]$ $= E[(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+2E[(Y-E(\hat{Y}))(E(\hat{Y})-\hat{Y})]]$ $= [(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+2(Y-E(\hat{Y}))E[(E(\hat{Y})-\hat{Y})]]$ $= [(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+2(Y-E(\hat{Y}))[E[E(\hat{Y})]-E[\hat{Y}]]$ $= [(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+2(Y-E(\hat{Y}))[E(\hat{Y})]-E[\hat{Y}]]$ $= [(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+2(Y-E(\hat{Y}))[0]$ $= [(Y-E(\hat{Y}))^2]+E[(E(\hat{Y})-\hat{Y})^2]+0$ **= [Bias²] + Variance**
![[Pasted image 20260416221101.png]]

**Polynomial Regression Example Question:** You are given a dataset where a model predicts house prices from house size. Students trained polynomial regression models of different degrees.

|**Model Degree**|**Training Error**|**Validation Error**|
|---|---|---|
|1|32|35|
|2|20|22|
|3|12|14|
|5|3|18|
|10|0|45|
||||

1. Which model is underfitting the data? - 1, 2, 3
    
2. Which model is overfitting the data? - 5, 10
    
3. Which model would you choose for deployment and why? - 3
    
4. Suppose we add 1000 more training samples. Which model's validation error would most likely improve? a. Degree 1 b. Degree 3 с. **Degree 10**
    Right now, it has too much power and not enough data, so it's "connecting the dots" and memorizing the noise. If you flood it with 1,000 new data points, it _can't_ memorize all of them. The sheer volume of data forces the wild, wobbly Degree 10 curve to smooth out and find the true underlying pattern, drastically dropping its validation error!

---

7. Regularization

Regularization in deep learning refers to a set of techniques and strategies employed to prevent overfitting in neural networks. Regularization methods help mitigate this problem by adding constraints or penalties to the neural network's learning process, discouraging it from fitting the noise in the training data and encouraging it to capture the underlying patterns.

Regularization: L1

L1 regularization adds a penalty term to the loss function based on the absolute values of the weights. It encourages sparsity in the model by pushing some of the weights to exactly zero.

- **Formula:** $L_{L1}(y,\hat{y}) = L(y,\hat{y}) + \lambda*\Sigma[w_i]$
    
- The value of $\lambda$ (lambda) determines the strength of the L1 regularization.
    
- A larger $\lambda$ leads to stronger regularization and sparser models.
    
- L1 regularization is often used for feature selection, as it automatically identifies and excludes less important features by setting their corresponding weights to zero.
    
- L1 regularization can help prevent overfitting by reducing the model's capacity and complexity.
![[Pasted image 20260416222021.png|240]]
    

Regularization: L2

L2 regularization adds a penalty term based on the squared values of the weights. It discourages large weight values and helps to smooth the learned function.

- **Formula:** $L_{L2}(y,\hat{y}) = L(y,\hat{y}) + \lambda*\Sigma w_i^2$
    
- The value of $\lambda$ (lambda) determines the strength of the L2 regularization.
    
- A larger $\lambda$ leads to stronger regularization and smaller weight magnitudes.
    
- L2 regularization is useful for preventing overfitting.
    
- It does not lead to sparsity in the model, meaning that all features (if not explicitly selected for removal) are retained and contribute to the predictions.
    
---

## **KNN & Naive Bayes**


**Part 1: K-Nearest Neighbors (KNN)**

### The Classification Problem

Let's once more discuss classification problem, based on given observations. The class that has the highest probability conditional to the given feature values will be considered as predicted class.

$\hat{y}=f(x)=argmax_{y\in\mathbb{Y}}P(Y=y|X=x)$

We have just found how to calculate the classifier - have we solved classification? **NO** When testing our classifier on new test data, we might encounter a different distribution of data and even new instances.

### Identifying the Flaws (PlayTennis Example)

Consider the following table:
![[Pasted image 20260417001056.png]]

**Classifier on training data:**

- $f(Overcast)=Yes$
    
- $f(Rain)=No$
    
- $f(Sunny)=No$
    

If we apply this classifier to the Testing Dataset, our predictions look like this:

- Day 3 (Overcast, Yes) -> Prediction: Yes
    
- Day 5 (Rain, Yes) -> Prediction: No
    
- Day 8 (Sunny, No) -> Prediction: No
    
- Day 9 (Sunny, Yes) -> Prediction: No
    
- Day 10 (Rain, Yes) -> Prediction: No
    
- Day 11 (Sunny, Yes) -> Prediction: No
    
- Day 13 (Overcast, Yes) -> Prediction: Yes
    

Our trained classifier had $4/7$ error rate on the test data - why so bad?

- unlucky with how the training and test data were split
    
- the dataset is small a bigger dataset might have allowed us to learn more successfully
    

**The Problem Gets Worse with More Features** If we add "Temp" to our data (Sunny/Hot, Rain/Mild, etc.), our trained classifier doesn't define what to predict for $f(Rain,Hot)=?$ If such instance occurs in unseen data then we would predict randomly.

If we add even more features (Day, Outlook, Temp, Humidity, Wind, PlayTennis): What would you predict for instance $x=$ (Sunny, Hot, Normal, Strong)? **Core Concept:** It is natural to hope that if two instances differ by only 1 feature then they are very likely to belong to the same class.

---

### **What is KNN?**

KNN (K-Nearest Neighbor) is a simple supervised learning algorithm used for classification and regression tasks.
![[Pasted image 20260417001913.png]]
**KNN: Regression** For each sample:
![[Pasted image 20260417002007.png]]

---

**KNN: Distance Functions**
![[Pasted image 20260417002145.png]]

### **Choosing 'K'**
![[Pasted image 20260417002310.png]]
    

---

**KNN: Weighting**

In K-Nearest Neighbors (KNN), weighting the neighbors allows you to account for the fact that some neighbors are more relevant to the prediction than others, especially when they are closer to the query point. Without weighting, KNN treats all neighbors equally, which may not always be optimal because closer neighbors often carry more important information than farther ones.

**Benefits of Weighting in KNN:**

- By giving more importance to closer neighbors, you can improve prediction accuracy, particularly in scenarios where distance is directly correlated with the relevance of the neighbor.
    
- Weighting can reduce the influence of distant or outlier points, which may belong to a different class or have values that do not accurately reflect the query point.
    
- Weighted KNN is more flexible as it allows the model to adapt more closely to the local structure of the data.
    
- _Formula example:_ o: $\frac{1}{d_{1}} + \frac{1}{d_{2}} + \frac{1}{d_{3}}$ x: $\frac{1}{d_{4}} + \frac{1}{d_{5}} \rightarrow$ predict o
    

---

**KNN: Normalization**

If features are on entirely different scales (e.g., Age 25 vs Salary 95,000), the larger numbers dominate the distance.

|**ID**|**Age**|**Salary**|**Loan decision**|
|---|---|---|---|
|1|25|95.000|Yes|
|2|60|90.000|No|
|3|26|92.000|?|

$distance(ID3,ID1)=\sqrt{(26-25)^{2}+(92000-95000)^{2}}=3000.00016667$ $distancc(ID3,ID2)=\sqrt{(26-60)^{2}+(92000-90000)^{2}}=2000.28897912$ $distance(ID3,ID1)>distance(ID3,ID2)\Rightarrow Loan~decision=No$

**Applying Min-Max Normalization:**

|**ID**|**Age***|**Salary***|**Loan decision**|
|---|---|---|---|
|1|0|1|Yes|
|2|1|0|No|
|3|0.0285|0.4|?|

$distance(ID3,ID1)=\sqrt{(0.028571-0)^{2}+(0.4-1.0)^{2}}=0.60067986651$ distance(ID3, ID2) = (0.028571-1.000000)2 + (0.4-0) = 1.05055904262 distance(ID3, ID2) > distance(ID3, ID1) => Loan decision = Yes

---

### **KNN Pros & Cons**

**PROS**

- easy to understand
    
- easy to implement
    
- zero training time
    
- works easily with multi-class datasets
    

**CONS**

- computationally expensive testing phase
    
- may not perform well when there is class imbalance
    
- with increasing dimensionality, since the difference in distances to the nearest and farthest neighbors becomes small, the performance is getting worse
    
---

**Part 2: Naive Bayes**

Naive Bayes is a family of classifiers based on Bayes' theorem assuming conditional independence of features in a class. NB assumes that the presence of any feature in a class is not related to the presence of any others.

For example, a fruit can be considered an apple if it is red, round and about 8 centimetres in diameter. Even if these traits depend on each other or on other traits, they contribute independently to the probability that this fruit is an apple. In connection with this assumption, the algorithm is called "naive".

**The Math:** According to Bayes' theorem, the probability of class y for an object with features $X_{1},X_{2},...,X_{n}$ is: $P(y|x_{1}x_{2}...x_{n})=\frac{P(y)P(x_{1}x_{2}...x_{n}|y)}{P(x_{1}x_{2}...x_{n})}$

Using the naive assumption of the conditional independence of features $X_{i}$ for a given y, we obtain $P(y|x_{1}x_{2}...x_{n})=\frac{P(y)\Pi P(x_{i}|y)}{P(x_{1}x_{2}...x_{n})}$

Since $P(x_{1}x_{2}...x_{n})$ is the same for all classes y, only the numerator affects the classification result. Therefore, the following classification rule can be used: $y=argmax~P(y)\Pi P(x_{i}|y)$

---
![[Pasted image 20260417003304.png]]

What we can say about the message: "Dear Friend"?

- P(Normal) P(Dear| Normal) P(Friend| Normal) = $0.67^{*}0.47^{*}0.29=0.09$
    
- P(Spam) P(Dear| Spam) P(Friend| Spam) = $0.33^{*}0.29^{*}0.14=0.01$
    
- => Normal
    

**The Zero-Frequency Problem:** 
![[Pasted image 20260417003442.png]]

**Why is it Naive?** The Naive Bayes algorithm is called "naive" because it makes a strong assumption of feature independence it assumes that all features in the dataset are independent of each other given the class label. In reality, this assumption is often unrealistic, as features are often correlated. However, despite this "naive" assumption, the algorithm performs surprisingly well in many practical applications, particularly in text classification and spam detection.

---

**Gaussian Naive Bayes**

The Gaussian Naive Bayes classifier assumes a Gaussian (normal) distribution for P(x | y). This approach is used to model the conditional probability of continuous features.

$P(x_{i}|y)=\frac{1}{\sqrt{2\pi\sigma_{y}^{2}}}exp(-\frac{(x_{i}-\mu_{y})^{2}}{2\sigma_{y}^{2}})$

**Example: Loves Peaky Blinders vs Doesn't love Peaky Blinders**

![[Pasted image 20260417004504.png]]

P(Loves Loves Peaky Blinders) $=8/(8+8)=0.5$ P(Doesn't love Loves Peaky Blinders) $=8/(8+8)=0.5$
**Initial Guess = Prior Probabilities**

**Testing a new instance:** Popcorn: 20, Coke: 500, Candy: 25 $P (Loves~Peaky~Blinders)*L(popcorn=20|Loves)^{*}L(coke=5oo|Loves)^{*}L(candy=25|Loves)=a$ really really small number <- **Underflow log function is used to avoid Underflow**

$log(P(Loves))*log(L(popcorn=20|Loves))*log(I(coke=500|Loves))*log(L(candy=25|Loves)) = -124$
$log(P(Doesn^{\prime}t)^{*}L(popcorn=20|Doesn^{\prime}t)^{*}L(coke=500|Doesn^{\prime}t)^{*}L(candy=25|Doesn't))=-48$

Based on those numbers, you would choose **"Doesn't love Peaky Blinders"**.

---

### **Naive Bayes Pros & Cons**

**PROS**

- Classification, including multiclass, is performed easily and quickly.
    
- When the independence assumption is met, it outperforms algorithms such as logistic regression while requiring less training data.
    
- Works with both categorical and continuous features.
    

**CONS**

- If some value of a categorical feature from the test set wasn't present in the training data set, then the model will assign a zero probability to this value and will not be able to make a prediction.
    
- The assumption of feature independence in reality, sets of completely independent features are rare.
    

---
This is a complete, word-for-word study guide based on your lecture slides. Since you are a visual learner, I have structured the information for high readability, added strategic image triggers to visualize the concepts, and built an interactive 3D simulation at the end so you can physically see how the "Kernel Trick" works!

---

## 8: Support Vector Machine**

## **Part 1: Support Vector Machine (SVM)**

### Introduction to Thresholds (1D Example)

Imagine we have data on a 1D line representing: **Not obese animals** and **Obese animals**.

We can choose a threshold based on observations:

- Less than the threshold = Not obese
    
- $More~than~the~threshold=Obese$
    

If a new observation appears and it is more than the threshold, we classify it as obese.

**The Problem with a Bad Threshold**

What if we place a threshold right next to the "not obese" animals?
![[Pasted image 20260417011032.png]]

It is much closer to the observations that are not obese. So this threshold doesn't work.

### The Maximal Margin Classifier (MMC)

Let's go back to the beginning and focus on the edges of two groups.

We will use the midpoint as a threshold. If a new observation appears and is closer to "not obese", it will be correctly classified based on this threshold.

- **Margin:** the shortest distance between the observations and the threshold.
    

Margin is the largest when the threshold is a midpoint. This classifier is called **Maximal Margin Classifier**.
![[Pasted image 20260417011107.png]]

**The Outlier Problem:**

What if our data look like this? We have an outlier which is not obese, but much closer to obese.

In this case, the MMC would be super close to the obese and really far from not obese.
![[Pasted image 20260417011131.png]]
If a new observation appears that is closer to obese, it will be classified as not obese based on the threshold.

**Drawback:** MMC is sensitive to outliers.

### The Soft Margin & Support Vectors

Let's put the threshold halfway between these two observations (ignoring the outlier). We will misclassify the outlier, but for a new observation, we will make a correct classification: obese.

Choosing a threshold which allows misclassifications, we can perform better on a new data.

- **Soft Margin:** When we allow misclassification, the distance between the observations and the threshold is called a Soft Margin.
    

We use Cross Validation to determine how many misclassifications and observations to allow inside of the soft margin to get the best result.

**Definition:** Support Vector Machine is a supervised machine learning algorithm most commonly used for solving classification problems and is also referred to as Support Vector Classification. The name Support Vector Machine comes from the fact that observations on the edge of the soft margin are called **Support Vectors**.

---

### Moving to 2D (Height & Weight)

What if we take into account not only weight but also height? Since each observation has 2 coordinates the data are now 2D.

When data is 2d the SVM is a line. And we can draw a SVM that separates the classes.

**The Math Behind the SVM Line:**

Let's define hyperlines H such that:
![[Pasted image 20260417011216.png]]
    
Formula for the distance from a point $(x_{\theta}y_{\theta})$ to a line $Ax_{0}+By_{0}+C=0$ is $Ax_{0}+By_{0}+C/\sqrt{A^{2}+B^{2}}$.

$H_{o}$ and $H_{1}$ are parallel lines, so the distance between $H_{o}$ and $H_{1}$ is $w^{*}x+b/||w||=1/||w||$.

So the distance (margin) between $H_{1}$ and $H_{2}$ is $2/||w||$.

**The Optimization Goal:**

The distance between $H_{1}$ and $H_{2}$ is $2/||w||$. In order to maximize the margin, we thus need to minimize $||w||$.

With the condition that are no data points between $H_{1}$ and $H_{2}$ :

- $w^{*}x_{i}+b\ge+1$ when $y_{i}=+1$
    
- $w^{*}x_{i}+b\le-1$ when $y_{i}=-1$
    

Can be combined into: $y_{i}(x_{i}^{*}w)\ge+1$

Minimization problem is solved using Lagrange Multipliers and Dual Problem.

**Why not Gradient Descent?**

How to find weights of the decision function? SVM optimization problem is a case of constrained optimization problem, and it is always preferred to use dual optimization algorithm to solve such constrained optimization problem. That's why we don't use gradient descent.

For a given dataset $\{(x_{i},y_{i})\}_{i=1}^{m}$, where $x_{i}$ are features and $y_{i}\in\{-1,1\}$ are labels, the SVM optimization problem is:

$min_{w,b}\frac{1}{2}||w||^{2}$

subject to: $y_{i}(w^{T}x_{i}+b)\ge1$ , Vi

$L(w,b,\alpha)=\frac{1}{2}||w||^{2}-\sum_{i}\alpha_{i}(y_{i}(w^{T}x_{i}+b)-1)$

---

## **Part 2: Kernel Trick**

### Non-linear Separability
![[Pasted image 20260417011513.png]]
    
Using a regular SVM we can't separate data. This type of data is called **linearly inseparable**.

**Solution (Adding a Dimension):**

Let's add y-axis. The y-axis coordinates will be the square of the dosages. The x-axis coordinates will be the dosages that we already observed.

Since each observation has 2 coordinates the data are now 2D. And we can draw a SVM that separates the classes. For a new example of dosage we could calculate y-axis, and the example will be correctly classified as not cured.

**General Rules for Inseparable Data:**

1. Start with data in a relatively low dimension
    
2. Move the data into a higher dimension
    
3. Find a Support Vector Classifier that separates the higher dimensional data into two classes.
    

> **Wait a minute**
> 
> Why we choose the square? Why not (π/3)*√Dosage?

### The Kernel Trick Explained

SVM works best when data is linearly separable. However, in many real-world problems, the data is not linearly separable.

For example: Non-linear Separability: Concentric Circles. We have data in 2D that looks like a circular pattern, where:

- Class 1 points are inside a circle.
    
- Class 0 points are outside.
    
    A straight line cannot separate these classes.
    

We could map the data to 3D using a transformation function, such as:

$\phi(x)=(x_{1},y_{2}x_{1}+y_{2})^{2}$

Now, the data becomes linearly separable in 3D. This transformation $\phi(x)$ is called a feature mapping.

**The Problem with Explicit Feature Mapping:**

If we explicitly compute $\phi(x)$ and perform SVM in a higher-dimensional space, it becomes computationally expensive, especially when using very high-dimensional transformations. The problem with moving data to higher dimensions (e.g., from 2D to 1.000D) is the "Curse of Dimensionality."

**Solution:**

The **Kernel trick** is a fundamental technique in machine learning, that allows algorithms to operate in a high-dimensional, implicit feature space without explicitly computing the coordinates of the data in that space. It relies on a specific mathematical property: many algorithms only care about the dot product (the similarity) between data points, not the actual coordinates of the points themselves.

Traditional way: Calculate $\phi(x)$, calculate $\phi(y)$, then find their dot product: $\phi(x)*\phi(y)$.

A **Kernel function** $K(x,y)$ is a "shortcut" function that calculates the dot product in that high-dimensional space using only the original low-dimensional coordinates.

$K(x,y)=\phi(x)\phi(y)$

This avoids explicitly computing $\phi(x)$ while still achieving the effect of transforming the data.

**Mercer's Theorem:**

The backbone of the trick is Mercer's Theorem. It states that any continuous, symmetric, positive semi-definite kernel function $K(x,y)$ can be expressed as a dot product in some high-dimensional feature space.

In simpler terms: as long as the kernel function follows certain mathematical rules, we are guaranteed that a corresponding high-dimensional space exists, even if we don't know exactly what that space looks like or how to define the mapping function $\phi(x).$

### Common Kernel Functions

**Polynomial Kernel**

$K(x,x^{\prime})=(\gamma x^{T}x^{\prime}+r)^{d}$

Introduces non-linearity by raising the dot product to a power d.

Parameters:

- γ: Scale factor (usually $\gamma>0$)
    
- r: Coefficient term.
    
- d: Degree of the polynomial.
    

**Radial Basis Function (RBF) Kernel**

$K(x,x^{\prime})=exp(-\gamma||{x-x'}||^{2})$

Parameters:

- γ: Scale factor (usually $\gamma>0$)
    

---

Here is the complete, word-for-word study guide based on your presentation. Because you are a visual learner, I have structured the text for maximum scannability and strategically inserted image tags so you can trigger visual diagrams for the most important concepts!

---

## **9. Decision Trees**

---

### **01 Classification Trees**

Decision Trees are a non-parametric supervised learning method used for classification and regression.

The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

**Will the game take place or not?**

1. If the weather is sunny and the humidity is normal, then yes.
    
2. If the weather is sunny, but the humidity is high, then no.
    
3. If the weather is rainy and with a strong wind, then no.
    
4. If the weather is rainy, but the wind is weak, then yes.
    
5. If the weather is cloudy, then yes.
    

**Tree Terminology**
![[Pasted image 20260417013109.png]]
    ![[Pasted image 20260417013138.png]]
    ![[Pasted image 20260417013149.png]]
    ![[Pasted image 20260417013202.png]]

**Decision Boundary**
![[Pasted image 20260417013238.png]]
**Algorithm Steps**

Decision Tree algorithm works in simpler steps:

1. **Starting at the Root:** The algorithm begins at the top representing the entire dataset.
    
2. **Asking the Best Questions:** It looks for the most important feature or question that splits the data into the most distinct groups.
    
3. **Branching Out:** Based on the answer to that question, it divides the data into smaller subsets, creating new branches. Each branch represents a possible route through the tree.
    
4. **Repeating the Process:** The algorithm continues asking questions and splitting the data at each branch until it reaches the final "leaf nodes," representing the predicted outcomes or classifications.
    

Now you must be thinking how do I know what should be the root node? What should be the decision node? When should I stop splitting?

**Classification Example: Loves Mr. Robot**

|**Loves Coding**|**Loves Coffee**|**Age**|**Loves Mr. Robot**|
|---|---|---|---|
|Yes|Yes|7|No|
|Yes|No|12|No|
|No|Yes|18|Yes|
|No|Yes|35|Yes|
|Yes|Yes|38|Yes|
|Yes|No|50|No|
|No|No|83|No|

**Evaluating Splits:**

If we split by "Coding" (True/False) versus "Coffee" (True/False), we get different mixtures of "Yes" and "No" answers in the leaves.

Leaves, containing mixture of classes are called Impure.

Since the "Coffee" tree contains more Pure leaves it seems better to classify data.

It would be perfect if we can quantify the difference between "Coding" and "Coffee" trees.

**Measuring uncertainty:** Good split if we are more certain about classification after split.

- Deterministic good (all true or all false)
    
- Uniform distribution bad
    
- What about distributions in between?
    

---

### **02 Entropy**

Entropy is nothing but the uncertainty in our dataset or measure of disorder.

**Example:** Suppose you have a group of friends who decides which movie they can watch together on Sunday.

There are 2 choices for movies, one is "Dune" and the second is "Avatar" and now everyone has to tell their choice. After everyone gives their answer we see that "Avatar" gets 4 votes and "Dune" gets 5 votes.

This is exactly what we call disorderness, there is an equal (somehow) number of votes for both the movies, and we can't really decide which movie we should watch.

It would have been much easier if the votes for "Dune" were 8 and for "Avatar" it was 2. Here we could easily say that the majority of votes are for "Dune" hence everyone will be watching this movie.

#### **Entropy Formula**

$E(S)=-p_{(+)}\log p_{(+)}-p_{(-)}\log p_{(-)}$

Here:

- $p_{+}$ is the probability of positive class
    
- $p_{-}$ is the probability of negative class
    
- S is the subset of the training example
    

Entropy basically measures the impurity of a node. A pure sub-split means that either you should be getting "yes", or you should be getting "no".

#### **Calculating Entropy Example**
![[Pasted image 20260417013738.png]]

**High Entropy:**

Variable has a uniform like distribution. Flat histogram. Values sampled from it are less predictable.

**Low Entropy:**

Distribution of variable has many peaks and valleys. Histogram has many lows and highs. Values sampled from it are more predictable.

---

### **03 Information Gain**

As mentioned earlier the goal is to decrease the uncertainty or impurity in the dataset, here by using the entropy we are getting the impurity of a particular node, we don't know if the parent entropy or the entropy of a particular node has decreased or not.

For this, a new metric called "Information gain" is used which tells us how much the parent entropy has decreased after splitting it with some feature.

Information Gain (IG) is the reduction in entropy achieved by partitioning a dataset according to a specific feature. It tells us how much "information" a feature provides about the target variable.

When building a tree, the algorithm tests every possible feature and chooses the one that results in the highest Information Gain.

**Formula:**

Information Gain $=E(Y)-E(Y|X)$

#### **Gym Example**

Example: Suppose our entire population has a total of 30 instances. The dataset is to predict whether the person will go to the gym or not. Let's say 16 people go to the gym and 14 people don't.

Now we have two features to predict whether he/she will go to the gym or not.

Feature 1 is "Energy" which takes two values "high" and "low".

Feature 2 is "Motivation" which takes 3 values "No motivation", "Neutral" and "Highly motivated".

**Testing Feature 1 (Energy):**
![[Pasted image 20260417013953.png]]

The entropy of the dataset will decrease by 0.37 if we make "Energy" as our root node.

**Testing Feature 2 (Motivation):**
![[Pasted image 20260417014304.png]]
"Energy" feature gives more reduction which is 0.37 than the "Motivation" feature. Hence we will select the feature which has the highest information gain and then split the node based on that feature.

#### **Continuous Features**

What if features are continuous? If the features are continuous, internal nodes may test the value of a feature against a threshold (e.g., > 75%, <= 75%, > 20, <= 20).

Binary tree, split on attribute X:

- One branch: $X<t$
    
- Other branch: $X\ge t$
    

Search through possible values of t. Seems hard!!!

1. Sort the dataset by the feature X.
    
2. For each pair of consecutive examples with different class labels, calculate the midpoint between their X values. These midpoints are candidate thresholds: $\theta=\frac{x_{i}+x_{i+1}}{2}$
    
3. For each candidate threshold $\theta$:
    
    - Split the dataset into two groups: $X\le\theta$ and $X>\theta$
        
    - Compute the information gain of the split.
        
    - Select the threshold that gives the highest information gain.
        

---

### **04 Pruning**

The biggest problem of decision trees is overfitting.

There is two possible solution:

1. **Pre-pruning** stops the tree from growing while it is still being built. You set constraints to cut off the growth before the tree reaches maximum complexity.
    
2. **Post-pruning** allows the tree to grow to its full, complex size first, and then it "trims" the branches that are least effective. This is generally considered more effective than pre-pruning because it allows the tree to see the full context of the data before deciding what to remove.
    

#### **Decision Trees: Pre-pruning**

**Max_depth parameter:** The more the value of max_depth, the more complex your tree will be. The training error will decrease if we increase the max_depth value but when our test data comes into the picture, we will get a very bad accuracy. Hence you need a value that will not overfit as well as underfit our data and for this.

**Min_samples_split parameter:** Specify the minimum number of samples required to do a spilt. For example, we can use a minimum of 10 samples to reach a decision. That means if a node has less than 10 samples then using this parameter, we can stop the further splitting of this node and make it a leaf node.

#### **Decision Trees: Post-pruning**

**Reduced Error Pruning:** Pruning the nodes (remove subtrees) if doing so does not increase the classification error on the validation set.

**Cost Complexity Pruning:** Pruning by iteratively removing branches with the weakest contribution to the overall accuracy. Calculate the cost complexity for each possible subtree.

Cost = Classification Error + a * Number of Leaves

Here, a is a regularization parameter that controls the trade-off between the complexity (size) of the tree and its fit to the training data.

---

### **05 Regression Trees**

Imagine we should develop a system to predict baseball player salary. Salary is color-coded from low (blue, green) to high (yellow, red).

Overall, the tree stratifies or segments the players into three regions of predictor space:

$R_{1}=\{X|Years<4.5\}$

$R_{2}=\{X|Years\ge4.5,Hits<117.5\}$

$R_{3}=\{X|Years\ge4.5,Hits\ge117.5\}$

- Years is the most important factor in determining Salary, and players with less experience earn lower salaries than more experienced players.
    
- Given that a player is less experienced, the number of Hits that he made in the previous year seems to play little role in his Salary.
    
- But among players who have been in the major leagues for five or more years, the number of Hits made in the previous year does affect Salary, and players who made more Hits last year tend to have higher salaries.
    

#### **Regression Trees: General idea**

We divide the feature space into J distinct and non-overlapping regions $R_{1},R_{2},...,R_{J}$.

For every observation that falls into the region $R_{i}$, we make same prediction, which is simply the mean of the response values for the training observations in Ri.

**Objective:** Find boxes $R_{1},R_{2},...,R_{J}$ that minimizes Residual Sum of Square (RSS).

$RSS=\sum_{i=1}^{J}\sum_{j\in R_{i}}(y_{j}-\hat{y_{R_{i}}})^{2}$

where $y_{R_{i}}$ is the mean response for the training in the i-th box.

1. We first select the feature $X_{i}$ and the cutpoint s such that splitting the feature space into the regions $\{X|X_{i}<s\}$ and $\{X|X_{i}\ge s\}$ leads to the greatest possible reduction in RSS.
    
2. Next, we repeat the process, looking for the best attribute and best cutpoint in order to split the data further so as to minimize the RSS within each of the resulting regions.
    
3. The process continues until a stopping criterion is reached; for instance, we may continue until no region contains more than five observations.
    

#### **Decision Trees Pros and Cons**

**PROS:**

- Works for numerical or categorical data and variables.
    
- Requires less data cleaning than other data modeling techniques.
    
- Can be used for feature selection.
    
- Easy to explain to those without an analytical background.
    

**CONS:**

- Affected by noise in the data.
    
- Not ideal for large datasets.
    
- Can disproportionately value, or weigh, attributes.
    
- Trees can become very complex when dealing with uncertainty and numerous linked outcomes.
    
- Prone to overtraining.
    
- When new training examples appear, updating the tree requires a complete build from scratch.
    

---

### **06 Random Forest**

Random Forest is a powerful and versatile ensemble algorithm that is primarily used for classification and regression tasks.

It works by constructing multiple decision trees during training and aggregating their predictions to make a more accurate and robust final prediction.

It is an extension of decision tree algorithms that enhances performance by reducing overfitting and increasing generalization.

|**Chest Pain**|**Good Blood Analysis**|**Arterial blockades**|**Weight**|**CVD**|
|---|---|---|---|---|
|No|No|No|125|No|
|Yes|Yes|Yes|180|Yes|
|Yes|Yes|No|210|No|
|Yes|No|Yes|167|Yes|

1. **Create a "bootstrapped" dataset.** Bootstrapped dataset is dataset with the same size as original one, containing samples from original dataset. Samples can repeat.
    
2. **Create a decision tree** using bootstrapped dataset and using a random subset of features at each step.
    
3. **Run new sample through each tree** from Random Forest and apply max-voting (or averaging in Regression Trees).
    

How to know that Random Forest works? Since we have a sample that hasn't been used during tree creation, it can be used for testing random forest.

---

### **1. Calculate Steps of Gradient Descent**

Gradient Descent is an optimization algorithm used to find the values of parameters (weights) that minimize a cost function (how wrong the model is).

**The Core Formula:**

$\theta_{new} = \theta_{old} - \alpha \frac{\partial J}{\partial \theta}$

- $\theta$: The weight or parameter you are updating.
    
- $\alpha$: The Learning Rate (how big of a step you take).
    
- $\frac{\partial J}{\partial \theta}$: The Gradient (the slope or derivative of the cost function at your current position).
    

**The Step-by-Step Process:**

1. **Initialize Weights:** Start with a random guess for your weights (e.g., $\theta = 0$).
    
2. **Calculate the Cost:** Evaluate how wrong your current model is using a Loss Function $J(\theta)$ (like Mean Squared Error).
    
3. **Calculate the Gradient:** Find the derivative of the cost function. This tells you the direction of the steepest ascent (uphill).
    
4. **Update the Weights:** Subtract the gradient multiplied by the learning rate from your current weights. (Subtracting moves you _downhill_).
    
5. **Repeat:** Loop through steps 2-4 until the cost stops decreasing (you reach convergence/the bottom of the curve).
    

---

### **2. Calculate and Analyse Quality Metrics**

You need to know how to judge your model. You cannot use the same metrics for predicting categories (Classification) as you do for predicting continuous numbers (Regression).

#### **Classification Metrics**

First, map your results on a **Confusion Matrix**:

- **True Positive (TP):** Predicted Yes, Actually Yes.
    
- **True Negative (TN):** Predicted No, Actually No.
    
- **False Positive (FP):** Predicted Yes, Actually No (Type I Error).
    
- **False Negative (FN):** Predicted No, Actually Yes (Type II Error).
    

|**Metric**|**Formula**|**When to use it / What it analyzes**|
|---|---|---|
|**Accuracy**|$\frac{TP + TN}{Total}$|Use when classes are perfectly balanced. Fails miserably on imbalanced data (e.g., predicting a rare disease).|
|**Precision**|$\frac{TP}{TP + FP}$|Use when **False Positives are costly**. (e.g., Spam filters. You don't want to send an important email to the spam folder).|
|**Recall (Sensitivity)**|$\frac{TP}{TP + FN}$|Use when **False Negatives are deadly**. (e.g., Cancer detection. It is better to over-diagnose than to miss a sick patient).|
|**F1-Score**|$2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$|The harmonic mean of Precision and Recall. Use when you have an **imbalanced dataset** and need a balance between Precision and Recall.|

#### **Regression Metrics**

| **Metric**                               | **Formula**                                                             | **When to use it / What it analyzes**                                                                                                          |
| ---------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **MAE (Mean Absolute Error)**            | $\frac{1}{n}  \sum \|y_i - \hat{y}_i\|$                                 | Easy to understand. Gives the average error in the same units as your target. Robust to outliers.                                              |
| **MSE (Mean Squared Error)**             | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$                                  | Heavily penalizes large errors because it squares them. Use when you want to strongly discourage large mistakes.                               |
| **RMSE (Root Mean Squared Error)**       | $\sqrt{MSE}$                                                            | Like MSE, but brings the unit back to the original scale of the target variable.                                                               |
| **$R^2$ (Coefficient of Determination)** | $1 - \frac{\text{Sum Squared Regression}}{\text{Total Sum of Squares}}$ | Measures how much of the variance in the target is explained by the model. **1.0** is perfect, **0.0** is no better than guessing the average. |

---

### **3. Choose Algorithms & Preprocessing Based on the Task**

When faced with a machine learning task, use this logic flow to make your decisions.

#### **Step A: Data Preprocessing**

- **Missing Data:**
    
    - _Drop rows:_ Only if the missing data is very small (e.g., < 5%).
        
    - _Impute (Mean/Median):_ For numerical data. Use Median if the data has extreme outliers.
        
    - _Impute (Mode):_ For categorical data.
        
- **Categorical Variables:**
    
    - _One-Hot Encoding:_ Use for nominal data with no order (e.g., Colors: Red, Blue, Green).
        
    - _Label Encoding:_ Use for ordinal data with a clear hierarchy (e.g., Sizes: Small, Medium, Large).
        
- **Feature Scaling:** (Crucial for KNN, SVM, and Gradient Descent models; not needed for Trees/Random Forests).
    
    - _Standardization (Z-score):_ Centers data around a mean of 0 and standard deviation of 1. Best default choice.
        
    - _Min-Max Normalization:_ Scales data between 0 and 1. Use when you need bounded values (e.g., image pixels).
        

#### **Step B: Algorithm Selection Cheat Sheet**

|**Task Type**|**Dataset Characteristics**|**Best Algorithm Choice**|**Why?**|
|---|---|---|---|
|**Classification**|Simple, linearly separable, need high interpretability|**Logistic Regression**|Easy to explain, outputs clear probabilities.|
|**Classification**|Text data, spam detection, word counts|**Naive Bayes**|Handles high-dimensional data well, mathematically assumes feature independence.|
|**Classification**|Complex, non-linear boundaries, high dimensionality|**SVM (with Kernel Trick)**|Kernels project data into higher dimensions to find clear margins.|
|**Class. / Regress.**|Mixed data types, need to understand feature importance|**Decision Trees**|Highly interpretable, handles categorical and continuous data easily. Prone to overfitting.|
|**Class. / Regress.**|Need high accuracy, worried about overfitting|**Random Forest**|Bagging technique reduces the variance (overfitting) of single decision trees.|
|**Class. / Regress.**|Tabular data, winning Kaggle competitions, highly complex|**Gradient Boosting / XGBoost**|Sequentially corrects errors of previous models. Highly accurate, but requires careful tuning of learning rate and depth to avoid overfitting.|

---


### **1. Example: Entropy & Information Gain**

Imagine you are trying to predict if a customer will buy a new video game. You have a dataset of **10 customers**.

- **6** bought the game (Yes)
    
- **4** did not buy it (No)
    

**Step A: Calculate the Parent Entropy (The initial disorderness)** The formula is: $E = -p_{(+)}\log_2 p_{(+)} - p_{(-)}\log_2 p_{(-)}$

- Probability of Yes ($p_+$) = 6/10 = 0.6
    
- Probability of No ($p_-$) = 4/10 = 0.4
    
- $E(Parent) = -(0.6)\log_2(0.6) - (0.4)\log_2(0.4)$
    
- $E(Parent) = -(0.6 \cdot -0.737) - (0.4 \cdot -1.322)$
    
- **$E(Parent) = 0.442 + 0.529 = 0.971$**
    
    _(An entropy of 0.971 is close to 1.0, meaning the dataset is highly mixed/impure)._
    

**Step B: Split the Data (e.g., by "Age Group")** Let's see what happens if we split these 10 customers into **Youth** and **Adults**.

- **Youth (4 people total):** 1 bought (Yes), 3 didn't (No).
    
    - $E(Youth) = -(1/4)\log_2(1/4) - (3/4)\log_2(3/4)$
        
    - **$E(Youth) = 0.811$**
        
- **Adults (6 people total):** 5 bought (Yes), 1 didn't (No).
    
    - $E(Adults) = -(5/6)\log_2(5/6) - (1/6)\log_2(1/6)$
        
    - **$E(Adults) = 0.650$**
        

**Step C: Calculate Information Gain** How much did this "Age" split help us clear up the mess? We take the Parent Entropy and subtract the _weighted average_ of the child entropies.

- Weighted Child Entropy = $(\frac{4}{10} \cdot 0.811) + (\frac{6}{10} \cdot 0.650)$
    
- Weighted Child Entropy = $0.324 + 0.390 = 0.714$
    
- **Information Gain** = $0.971 (\text{Parent}) - 0.714 (\text{Children}) =$ **$0.257$**
    

Conclusion: Splitting by Age gives us an Information Gain of 0.257. If another feature (like "Has Console") gives an Information Gain of 0.400, the Decision Tree will choose "Has Console" as the root node instead!

---

### **2. Example: Gradient Descent (One Step)**

Let's optimize a very simple cost function: $J(w) = w^2$.

Our goal is to find the weight ($w$) that makes this cost as close to $0$ as possible. The derivative (gradient) of $w^2$ is $2w$.

- **Starting Guess:** $w = 5$
    
- **Learning Rate ($\alpha$):** $0.1$
    
- **Current Cost:** $J(5) = 5^2 = 25$
    

**The Update Formula:** $w_{new} = w_{old} - \alpha \cdot \text{Gradient}$

1. **Find the Gradient:** $2 \cdot w_{old} \rightarrow 2 \cdot 5 = 10$. (The slope is steep and positive).
    
2. **Calculate the Step:** Multiply gradient by learning rate $\rightarrow 10 \cdot 0.1 = 1$.
    
3. **Update the Weight:** $w_{new} = 5 - 1 = 4$.
    

_Result:_ In just one step, our weight moved from 5 to 4. Our new cost is $J(4) = 4^2 = 16$. We successfully walked downhill from a cost of 25 to 16!

---

### **3. Example: Calculating Quality Metrics**

You built a machine learning model to detect **Spam Emails**. You test it on 100 emails (25 are actually spam, 75 are actually safe).

**Your Confusion Matrix Results:**

- **True Positive (TP):** 20 (You caught 20 spam emails).
    
- **True Negative (TN):** 70 (You let 70 good emails through).
    
- **False Positive (FP):** 5 (You accidentally sent 5 good emails to the spam folder).
    
- **False Negative (FN):** 5 (You let 5 spam emails sneak into the inbox).
    

**Let's analyze the quality:**

- **Accuracy:** $\frac{20 + 70}{100} =$ **$90\%$**. (Sounds great, but let's dig deeper).
    
- **Precision:** $\frac{TP}{TP + FP} = \frac{20}{20 + 5} = \frac{20}{25} =$ **$80\%$**. (When your model claims an email is spam, it is only right 80% of the time. Those 5 False Positives could be important missed emails!).
    
- **Recall:** $\frac{TP}{TP + FN} = \frac{20}{20 + 5} = \frac{20}{25} =$ **$80\%$**. (Your model successfully found 80% of all the real spam in the dataset).
    
- **F1-Score:** $2 \cdot \frac{0.80 \cdot 0.80}{0.80 + 0.80} =$ **$0.80$**.
    

---


![[Pasted image 20260417021712.png]]
![[Pasted image 20260417021818.png]]




![[Pasted image 20260417022759.png]]

![[Pasted image 20260417022728.png]]






























































 
























