![[Pasted image 20260801171224.png]]

![[Pasted image 20260801173116.png]]

Let's look at the **entire end-to-end story of AlexNet**.

AlexNet takes a raw $227 \times 227 \times 3$ RGB image at one end and outputs a probability distribution across **1,000 object classes** at the other.

## The Big Picture Pipeline

AlexNet is divided into two major functional zones across its **8 trainable layers**:

Plaintext

```
  [ Raw RGB Image ] ──► (227 x 227 x 3)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION ZONE                    │
│                                                         │
│  Conv1 + ReLU + LRN + MaxPool1  ─► (27 x 27 x 96)     │
│  Conv2 + ReLU + LRN + MaxPool2  ─► (13 x 13 x 256)    │
│  Conv3 + ReLU                   ─► (13 x 13 x 384)    │
│  Conv4 + ReLU                   ─► (13 x 13 x 384)    │
│  Conv5 + ReLU + MaxPool5        ─► (6 x 6 x 256)      │
└─────────────────────────────────────────────────────────┘
           │
           ▼ [ Flattening: 6 x 6 x 256 ──► 9,216 Vector ]
           │
┌─────────────────────────────────────────────────────────┐
│              CLASSIFICATION ZONE                        │
│                                                         │
│  FC6 + ReLU + Dropout (0.5)     ─► (4,096 Vector)     │
│  FC7 + ReLU + Dropout (0.5)     ─► (4,096 Vector)     │
│  FC8 (Linear Output)            ─► (1,000 Logits)     │
└─────────────────────────────────────────────────────────┘
           │
           ▼
   [ Softmax Function ] ──► 1,000 Class Probabilities
```

## Step-by-Step: What Happens Stage by Stage?

### 1. Low-Level Feature Extraction (`conv1` & `conv2`)

- **`conv1`:** Uses large $11 \times 11$ filters with stride 4. It acts like a coarse visual scanner, looking for basic primitives in the image: **oriented edges, blobs of color, and high-contrast lines**.
    
- **`conv2`:** Takes these edge maps and combines them into slightly more complex geometric patterns (like corners, textures, and simple grids).
    
- **Local Response Normalization (LRN):** Applied after both `conv1` and `conv2`. LRN dampens neurons that have super high activations so they don't dominate neighboring channels (mimicking lateral inhibition in real biological eyes).
    

### 2. Deep Feature Hierarchy (`conv3`, `conv4`, `conv5`)

Unlike earlier layers, `conv3`, `conv4`, and `conv5` are stacked **directly on top of each other** without max-pooling layers in between:

- **`conv3` & `conv4`:** Combine simpler shapes into object parts (e.g., eyes, wheels, dog noses, fabric patterns).
    
- **`conv5`:** Assembles these parts into high-level object representations (e.g., full animal faces, car bodies).
    
- **`maxpool5`:** Shrinks the spatial dimension down to a compact $6 \times 6$ grid while preserving the strongest activated features.
    

### 3. Spatial Flattening (The Bridge)

Before moving into the dense classifier, the 3D output tensor of `maxpool5` ($6 \times 6 \times 256$) is unrolled row-by-row into a **1D vector of length 9,216**:

$$\text{Flattened Vector} = [x_1, x_2, x_3, \dots, x_{9216}]$$

This destroys spatial grid locations and turns spatial features into abstract numerical concept scores.

### 4. Dense Reasoning & Classification (`fc6`, `fc7`, `fc8`)

- **`fc6` & `fc7`:** These layers act as decision-making networks. They take the 9,216 feature scores and mix them globally to build abstract concepts (e.g., _"If feature #102 [whiskers] AND feature #804 [fur] ARE present, boost class 'Cat'"_).
    
- **`fc8`:** Reduces the 4,096 features down to exactly **1,000 numbers** (called **logits** $z_1, z_2, \dots, z_{1000}$), one for each target class in the ImageNet dataset.
    

### 5. Final Output: Softmax & Loss Calculation

To turn raw logits into human-readable probabilities, the 1,000 output numbers are passed through the **Softmax** function:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}$$

- Every output value $p_i$ is strictly between $0$ and $1$.
    
- The sum of all 1,000 output probabilities equals **1.0** ($100\%$).
    

During training, the predicted probability distribution is compared against the ground-truth label using **Cross-Entropy Loss**:

$$\mathcal{L} = -\log(p_{\text{correct}})$$

Gradients are calculated from this loss and sent **backward** through all 62.3 million parameters via Backpropagation to adjust the weights.

## Key Innovations That Made AlexNet Revolutionary

AlexNet was the breakout paper (2012) that launched modern deep learning. Its success relied on four key design choices:

### 1. ReLU Instead of Tanh

Before AlexNet, models used $\tanh(x)$ or Sigmoid activations. AlexNet introduced **ReLU** ($\text{ReLU}(x) = \max(0, x)$), which solved the vanishing gradient problem and allowed the network to train **6x faster**.

### 2. Overlapping Max Pooling

Instead of standard non-overlapping pooling ($K=2, S=2$), AlexNet used $3 \times 3$ pooling windows with Stride 2 ($K=3, S=2$). The overlapping windows made it harder for the model to overfit.

### 3. Dropout (0.5)

In `fc6` and `fc7`, **50% of the neurons were randomly zeroed out** during each training forward pass. This forced the network to learn redundant representations rather than relying on specific individual neurons.

### 4. Data Augmentation

To prevent its 62 million parameters from memorizing the training images, AlexNet generated training samples on the fly:

- Random $227 \times 227$ crops from $256 \times 256$ images.
    
- Horizontal image reflections (flips).
    
- PCA color augmentation (altering image color intensities).