- [[#Splitting the Feature Space|Splitting the Feature Space]]
- [[#Perceptron Algorithm|Perceptron Algorithm]]
- [[#Support Vector Machines (SVM)|Support Vector Machines (SVM)]]
- [[#Kernel Functions|Kernel Functions]]
- [[#Decision Boundary & Scoring Function|Decision Boundary & Scoring Function]]
- [[#Hard Margin Optimization|Hard Margin Optimization]]
- [[#Soft Margin Classification & Slack Variables|Soft Margin Classification & Slack Variables]]
- [[#The Kernel Trick|The Kernel Trick]]

## Splitting the Feature Space

What are some other ways we can split the feature space? Other than decision boundaries:

We can simply draw a straight line.

$$w^T x + b$$

Given a feature vector and a weight vector:

- $\vec{x} = (x_1, x_2, \dots, x_n)$
    
- $\vec{w} = (w_1, w_2, \dots, w_n)$
    

The linear equation is:

$$y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \vec{w}^T \vec{x} + b$$

Classification based on the output score:

- $\vec{w}^T \vec{x} + b > 0 \rightarrow \circ$ (Class 1)
    
- $\vec{w}^T \vec{x} + b = 0 \rightarrow ?$ (On the boundary)
    
- $\vec{w}^T \vec{x} + b < 0 \rightarrow \Delta$ (Class 2)
    

**Note:** Not all datasets are linearly separable!

---

## Perceptron Algorithm

How can we find a good line to separate? $\rightarrow$ **Perceptron Algorithm**.

- _Armenian Note:_ սովորում ա սխալների վրա, ի տարբերություն KNN-ի կամ Decision Trees. _(It learns from mistakes, unlike KNN or Decision Trees)._
    

**Hypothesis Function:**

$$h(\vec{x}_i) = \text{sign}(\vec{w}^T \vec{x}_i + b)$$

**Sign Function Definition:**

$$\text{sgn}(x) = \begin{cases} 1, & x > 0 \\ 0, & x = 0 \\ -1, & x < 0 \end{cases}$$

- The weight vector ($\vec{w}$) that defines the hyperplane is **always perpendicular** to the decision boundary.
    
- The decision boundary (hyperplane $H$) is defined as: $H = \{x : \text{dot}(x, w) + b = 0\}$
    
- **Goal:** Given inputs ($x$'s), find $w$ and $b$.
    

### Evaluating Correct Classifications

To simplify notation, we can absorb the bias into the vector:

$\vec{w} = (b, w_1, w_2, \dots, w_n)$

$\vec{x} = (1, x_1, x_2, \dots, x_n)$

$\Rightarrow y = \vec{w}^T \vec{x}$

- _Եթե կետը ճիշտ ենք դասակարգել, ապա իր $y_i \cdot \vec{w}^T \vec{x}_i > 0$ _(If we classified the point correctly, then its $y_i \cdot \vec{w}^T \vec{x}_i > 0$)_.
    

**Breakdown:**

- $\vec{w}^T \vec{x}_i > 0 \Rightarrow$ կետը գծից վերև ա _(point is above the line)_ $\Rightarrow \hat{y}_i = 1$
    
    - Therefore, $y_i \cdot \vec{w}^T \vec{x}_i > 0$
        
- $\vec{w}^T \vec{x}_i < 0 \Rightarrow$ կետը գծից ներքև ա _(point is below the line)_ $\Rightarrow \hat{y}_i = -1$
    
    - Therefore, $y_i \cdot \vec{w}^T \vec{x}_i > 0$
        

### Perceptron Architecture & Bias

**Structure:**

1. **Inputs:** $x_1, x_2, \dots, x_n$
    
2. **Weights:** $w_1, w_2, \dots, w_n$
    
3. **Weighted Sum:** $\Sigma \rightarrow w^T x_i + b$
    
4. **Activation Function:** $\int \rightarrow h(x_i) = \text{sign}(w^T x_i + b)$ (Anything $>0 \rightarrow +1$, $<0 \rightarrow -1$)
    
5. **Output (binary):** $\hat{y} \in \{1, -1\}$
    

**Bias ($b$):**

Is a constant value added to the weighted sum to shift the decision boundary. It allows the perceptron to classify correctly even when all input features are zero. Bias ensures the model is not forced to pass the decision boundary through the origin $(0,0)$.

### Perceptron Learning Rule (Algorithm)

Given training data $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$:

1. Let $w \leftarrow [0, 0, \dots, 0]$
    
2. **Repeat:**
    
    - Let $\Delta \leftarrow [0, \dots, 0]$ _(սխալներ / errors)_
        
    - for $i = 1, \dots, n$ do
        
        - if $y^{(i)}(\vec{w}^T x^{(i)}) \le 0$ _(սխալ ենք գուշակել / we guessed incorrectly)_
            
            - $\Delta \leftarrow \Delta + y^{(i)}x^{(i)}$
                
    - $\Delta \leftarrow \Delta / n$ _(սխալների միջինը / average of errors)_
        
    - $w \leftarrow w + \alpha \Delta$ _(where $\alpha$ is the learning rate / հիպերպարամետր)_
        
3. **Until** $||\Delta||_2 < \epsilon$
    

- Թեորեմ: Եթե դասերն բաժանելի են, ուրեմն ինչ-որ վերջավոր քանակով քայլերից հետո կգտնի լուծում այս ալգորիթմով: Եթե չէ $\rightarrow$ անվերջ ցիկլ։ __

- Էս ալգորիթմը չունի ժամանակային բարդություն, զուտ սկալյար արտադրյալներ ա անում։ 
---

## Support Vector Machines (SVM)

**The Problem with Perceptrons:**

Հարցնում ենք, որ եթե գիծ կա ուրեմն կարանք գտնենք, բայց կարող ա շատ գծեր կան, որոնք բաժանում են, ոնց գտնենք լավագույնը: $\rightarrow$ **SVM**


Վերցնենք էն գիծը, որ էդ գծին ամենամոտ կետերի հեռավորությունը գծից ամենամեծն է:


### Support Vectors and Margin

- **Support Vectors:** Էն վեկտորները, որոնք ամենամոտ կետերով են կառուցվում, դրանով էլ margin ա սահմանվում։ _(Those vectors that are constructed by the closest points, and by that the margin is defined.)_
    
- **Distance calculation:** Distance = $\frac{\text{Total score needed}}{\text{Score gained per step}} = \frac{2}{||w||}$
    
- $w$ is the gradient of $wx + b$.
    

### Proof: 1 Unit Step in the Direction of $w$

Let's analyze what taking a step does to our function score:

- **Starting point:** $x_{\text{start}}$ (where $w^T x_{\text{start}} + b = 0$, starting on the boundary line)
    
- **Unit vector:** $\frac{w}{||w||}$
    
- **Direction to walk:** $\frac{w}{||w||}$ (unit vector)
    
- **New location:** $x_{\text{new}} = x_{\text{start}} + \frac{w}{||w||}$
    

Let $f(x) = w^T x + b$:

$$\text{New score} = w^T \left(x_{\text{start}} + \frac{w}{||w||}\right) + b$$

$$= (w^T x_{\text{start}} + b) + \left(w^T \frac{w}{||w||}\right)$$

Since $x_{\text{start}}$ is on the boundary, $(w^T x_{\text{start}} + b) = 0$:

$$= 0 + \frac{w^T w}{||w||} = \frac{||w||^2}{||w||} = ||w||$$

**Conclusion:** Taking 1 physical step literally adds $||w||$ to my score.

### SVM vs KNN Weights

BTW KNN-ում մենք օգտագործում էինք շատ կետեր, իսկ SVM-ում մի քանի հատ, զուտ որ support vector-ներ ունենանք:

A dot product is just a mathematical way to measure **similarity**.

$$w = \sum_i \alpha_i y_i x_i$$

Where:

- $\alpha_i$: An "importance" weight. For normal data points, $\alpha = 0$ (they don't matter). For support vectors, $\alpha > 0$.
    
- $y_i$: The class label of that vector (+1 or -1).
    
- $x_i$: The coordinates of a Support Vector.
    

---

## Kernel Functions

Examples of kernel functions:

- **Linear:** $k(x_i, x_j) = x_i^T x_j$ _(սկալյար արտադրյալ / dot product)_
    
- **Polynomial of power $p$:** $k(x_i, x_j) = (1 + x_i^T x_j)^p$
    
- **Gaussian (radial-basis func.):** $k(x_i, x_j) = e^{-\frac{||x_i - x_j||^2}{2\sigma^2}}$
    

**Weight Update Reminder:**

$w$ - պարամետր, որը perceptron-ը սովորելուց update ա անում ու հիպերպարամետրեր՝ $\alpha$ - learning rate.

$$w_1 = w_1 + \alpha(\hat{y} - y)x$$

_(Where $\hat{y}$ is the predicted output, and the difference represents the error/սխալվել ենք)_

## Decision Boundary & Scoring Function

When classifying data, the **decision boundary** is the line where every point on that line has a score of exactly $0$.

To make the distance from the origin to the plane $0$ (i.e., shifting the boundary to pass through the origin), we need to offset it by subtracting $\frac{b}{||w||}$.

The scoring function can be represented using the support vectors:

$$f(x) = \sum_i \alpha_i y_i (x_i^T x) + b$$

- **$f(x)$:** The scoring function.
    
- **$\sum_i \alpha_i y_i x_i^T$:** This entire term represents our weight vector $w$.
    
- **$x$:** The new datapoint we need to classify.
    

---

## Hard Margin Optimization

**Goal:** Find $w$ and $b$ to maximize the margin $\frac{2}{||w||}$.

**Subject to (s.t.):**

- $w^T x_i + b \ge 1$ for $y_i = +1$
    
- $w^T x_i + b \le -1$ for $y_i = -1$
    
- _Կետերը կլասիֆիկացված են ճիշտ for $i = \overline{1, N}$ _(The points are correctly classified for $i = \overline{1, N}$)_.
    

Maximizing $\frac{2}{||w||}$ is mathematically equivalent to minimizing $||w||^2$.

**Equivalent Formulation:**

Find $w$ and $b$ such that:

$$\min_w ||w||^2$$

Subject to: $y_i(w^T x_i + b) \ge 1$ for $i = \overline{1, N}$

_(Note: $||w||^2 = w^T w$)_

- _Գտնել են մինիմում վեկտորի երկարությունը $||w||$, որի համար կետերը կլասիֆիկացված են ճիշտ։ _(Find the minimum vector length $||w||$ for which the points are classified correctly.)_
    
- Գտնել էն $w$ հարթությունը (ուղիղը), որի համար $||w||^2$-ն ամենափոքրն ա և բոլոր կետերը ճիշտ են կլասիֆիկացված: _(Find the $w$ plane (line) for which $||w||^2$ is the smallest and all points are correctly classified.)_
    

---

## Soft Margin Classification & Slack Variables

Hard margin optimization strictly requires all points to be correctly classified, essentially assigning an infinite error to mistakes.

- Բայց մեզ դեռ չի տալիս որ կետերը լինեն ճիշտ կլասիֆիկացված, այսինքն սխալներին infinite error տալ, դրա համար մենք իրանց տալիս ենք ինչ-որ սխալներ, բայց ոչ infinite՝ margin-ը հնարավորինս մեծացնել, սխալ կետերին պատժել: _(But this still doesn't guarantee points are perfectly classified; instead of giving infinite error to mistakes, we allow some errors but not infinite—we maximize the margin as much as possible while punishing the wrong points.)_
    

To allow this, we introduce new variables for each point called **Slack Variables** ($\xi_n$):

- For data points that are on or inside the correct margin $\Rightarrow \xi_n = 0$
    
- For other points $\Rightarrow \xi_n = |t_n - y(x_n)|$
    
    _(where $t_n$ is the target value $\in \{1, -1\}$)_
    

**Conditions for $\xi_n$:**

- $\xi_n = 0 \Rightarrow$ Point is strictly on (or outside) the correct margin.
    
- $\xi_n \in (0, 1] \Rightarrow$ Ճիշտ $t=1$-երի վրա _(Correctly classified, but inside the margin area)_.
    
- $\xi_n > 1 \Rightarrow$ Point is misclassified.
    

**New Classification Constraint:**

$$t_n y(x_n) \ge 1 - \xi_n$$

### The Soft Margin Objective Function

_(Թույլ է տալիս կետերը սխալվեն)_

We must find the parameters $w$ and $b$ that minimize the following:

$$\min_{w,b} \left( C \sum_{n=1}^N \xi_n + \frac{1}{2} ||w||^2 \right)$$

- **$C$:** A hyperparameter.
    
- **$C \sum_{n=1}^N \xi_n$:** The penalty term. Ուշադրություն դարձնենք խնդրի լուծման վրա, ինչքան թույլ ենք տալիս որ սխալվեն _(Controls how much attention we pay to misclassifications / how much we allow them to make mistakes)_.
    
- **$\frac{1}{2} ||w||^2$:** The margin maximization term (we want this to be the smallest).
    
- _Armenian Note:_ Օպտիմիզացվող պարամետրերը $w$ ու $b$-ն են: _(The parameters being optimized are $w$ and $b$.)_
    

### The Role of Hyperparameter C

- Ինչքան $C$-ն մեծացնենք էնքան կձգտի բոլոր կետերը ճիշտ կլասիֆիկացնել, նույնիսկ եթե շատ վատ գիծ գծվի:
- 
- **Hard margin classifier:** $C \rightarrow \infty$
    
- Եթե ճիշտ ենք կլասիֆ. բոլոր կետերը $\Rightarrow \sum_{n=1}^N \xi_n = 0 \Rightarrow$ հետ ենք գալիս հին խնդրին:_
    
- **Goal:** Margin-ը լինի բավականին մեծ, misclassified կետերն էլ քիչ: ($C$-ն պարամետր ա) __
    

---

## The Kernel Trick

If data is not linearly separable, map the data onto a higher dimension!

Example Transformation ($\phi$): $x \rightarrow x^2$ or more generally, $\phi: x \rightarrow \phi(x)$.

- Էն ֆունկցիան, որը վերցնում ա որպես input original տիրույթից վեկտորներ և վերադարձնում ա էդ վեկտորների սկալյար արտադրյալը նոր տարածության մեջ կոչվում ա **kernel function**: 
    

$$k(x, y) = \phi(x)^T \phi(y)$$

_(Where $\phi$ is a vector mapping function)._

**More formally:** $k$ is a kernel if there exists $\phi : X \rightarrow \phi(x)$ such that for all $x, y \in X$, the inner product becomes $k(x, y) = \phi(x)^T \phi(y)$.

### Example: Showing the Implicit Mapping

Standard inner product: $k(x_i, x_j) = x_i^T x_j$

If every datapoint is mapped into high-dimensional space via some transformation $\phi : x \rightarrow \phi(x)$, the inner product becomes: $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$.

Let's take a polynomial kernel as an example:

- $x = [x_1, x_2]$
    
- $k(x_i, x_j) = (1 + x_i^T x_j)^2$
    

**Need to show that $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$:**

$$k(x_i, x_j) = (1 + x_i^T x_j)^2 = 1 + (x_{i1}x_{j1} + x_{i2}x_{j2})^2 + 2x_{i1}x_{j1} + 2x_{i2}x_{j2}$$

$$= 1 + x_{i1}^2x_{j1}^2 + 2x_{i1}x_{j1}x_{i2}x_{j2} + x_{i2}^2x_{j2}^2 + 2x_{i1}x_{j1} + 2x_{i2}x_{j2}$$

We can factor this into the dot product of two vectors:

$$= [1, x_{i1}^2, \sqrt{2}x_{i1}x_{i2}, x_{i2}^2, \sqrt{2}x_{i1}, \sqrt{2}x_{i2}]^T \cdot [1, x_{j1}^2, \sqrt{2}x_{j1}x_{j2}, x_{j2}^2, \sqrt{2}x_{j1}, \sqrt{2}x_{j2}]$$

$$= \phi(x_i)^T \phi(x_j)$$

**Where:**

$$\phi(x) = [1, x_1^2, \sqrt{2}x_1x_2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2]^T$$

> **Key Takeaway:** So, we don't compute each $\phi(x)$ explicitly. We just implicitly map data to a high-dim space with dot products using the kernel function.

### Examples of Kernel Functions

- **Linear:** $k(x_i, x_j) = x_i^T x_j$ _(սկալյար արտադրյալ / dot product)_
    
- **Polynomial of power $p$:** $k(x_i, x_j) = (1 + x_i^T x_j)^p$
    
- **Gaussian (Radial-Basis Function):** $k(x_i, x_j) = e^{-\frac{||x_i - x_j||^2}{2\sigma^2}}$
    

**Weight Update Rule (Perceptron Reminder):**

$w$ is the parameter that the perceptron updates while learning, and $\alpha$ is the learning rate hyperparameter.

$$w_1 = w_1 + \alpha(\hat{y} - y)x$$

_(Where $\hat{y} - y$ represents the error / սխալվել ենք)_
