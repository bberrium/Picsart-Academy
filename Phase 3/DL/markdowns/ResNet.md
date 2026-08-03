Let's look at the entire end-to-end story of **ResNet** (specifically **ResNet-18**).

ResNet18 takes a raw $224 \times 224 \times 3$ RGB image at one end (or $227 \times 227 \times 3$) and outputs a probability distribution across **1,000 object classes** at the other.

## The Big Picture Pipeline

ResNet18 is divided into two major functional zones across its **18 trainable weight layers** (1 initial conv layer + 8 residual blocks with 2 conv layers each + 1 final fully connected layer):

```
  [ Raw RGB Image ] ──► (224 x 224 x 3)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION ZONE                    │
│                                                         │
│  Stem: Conv1 (7x7, s=2) + MaxPool (3x3, s=2) ──► (56 x 56 x 64)│
│                                                         │
│  Layer 1: 2x Residual Blocks (64-d)          ──► (56 x 56 x 64)│
│  Layer 2: 2x Residual Blocks (128-d, s=2)    ──► (28 x 28 x 128)│
│  Layer 3: 2x Residual Blocks (256-d, s=2)    ──► (14 x 14 x 256)│
│  Layer 4: 2x Residual Blocks (512-d, s=2)    ──► (7 x 7 x 512)  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              CLASSIFICATION ZONE                        │
│                                                         │
│  Global Average Pooling (7x7)   ─► (1 x 1 x 512 Vector) │
│  FC (Linear Output Layer)       ─► (1,000 Logits)       │
└─────────────────────────────────────────────────────────┘
           │
           ▼
   [ Softmax Function ] ──► 1,000 Class Probabilities
```

## Step-by-Step: What Happens Stage by Stage?

### 1. The Stem Stage (`conv1` & `maxpool`)

Unlike VGG, which processes raw images slowly with small strides, ResNet starts with a rapid spatial reduction stem:

- **`conv1`:** Uses a large $7 \times 7$ kernel with **Stride 2** to extract initial low-level visual primitives while immediately halving spatial dimensions.
    
- **`maxpool`:** A $3 \times 3$ max pooling window with **Stride 2** cuts the spatial dimensions in half again down to $56 \times 56$.
    

### 2. Deep Residual Learning (Layer 1 through Layer 4)

Instead of standard sequential convolutions, ResNet structures its convolutional processing into **Residual Blocks**:

- **Layer 1 (Blocks 1 & 2):** Keeps spatial dimensions at $56 \times 56$ with 64 channels. Identity shortcuts skip across every 2 convolutional layers directly ($\mathbf{x} \rightarrow \mathcal{F}(\mathbf{x}) + \mathbf{x}$).
    
- **Layer 2 (Blocks 3 & 4):** Downsamples spatial size to $28 \times 28$ and doubles depth to 128 channels using a stride of 2. Uses a **Projection Shortcut** on the transition block to match dimensions.
    
- **Layer 3 (Blocks 5 & 6):** Downsamples to $14 \times 14$ and expands depth to 256 channels.
    
- **Layer 4 (Blocks 7 & 8):** Downsamples to $7 \times 7$ and expands depth to 512 channels. High-level abstract concepts are processed through residual updates.
    

### 3. Global Average Pooling (The Efficiency Bridge)

Unlike AlexNet and VGG, which use parameter-heavy Dense Fully Connected layers (`fc6` and `fc7` with over 100M+ parameters), ResNet replaces them with **Global Average Pooling (GAP)**:

- A $7 \times 7$ average pooling window computes the spatial mean of each $7 \times 7$ feature map.
    
- Reduces the $(7 \times 7 \times 512)$ tensor directly into a compact **1D vector of length 512**.
    

$$\text{GAP Vector} = [a_1, a_2, \dots, a_{512}]$$

This drastically reduces the model's total parameter count and acts as a strong regularizer against overfitting.

### 4. Dense Classifier (`fc`)

- **`fc`:** A single linear layer that connects the 512 average features directly to **1,000 output logits** ($z_1, z_2, \dots, z_{1000}$), representing the ImageNet classes.
    

### 5. Final Output: Softmax & Loss Calculation

Raw output logits are converted into normalized probabilities via **Softmax**:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}$$

- Every output value $p_i$ lies in $[0, 1]$.
    
- The sum of all 1,000 probabilities equals **1.0** ($100\%$).
    

During training, predictions are compared with ground-truth target classes using **Cross-Entropy Loss**:

$$\mathcal{L} = -\log(p_{\text{correct}})$$

Gradients flow back uninterrupted through **both** the convolutional paths and the shortcut identity connections via Backpropagation to adjust weights across all **~11.5 million parameters**.

## Key Innovations That Made ResNet Revolutionary

ResNet (2015 by Kaiming He et al., Microsoft Research) solved the **degradation problem** in deep neural networks, enabling networks to scale from 16 layers up to 152+ layers.

### 1. Residual Learning Paradigm ($\mathcal{F}(\mathbf{x}) + \mathbf{x}$)

Instead of forcing a stack of layers to directly fit an underlying mapping $\mathcal{H}(\mathbf{x})$, ResNet reformulates the target mapping into a **residual mapping**:

$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} \implies \mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

If a step needs an identity transformation ($\mathcal{H}(\mathbf{x}) = \mathbf{x}$), it is mathematically easier for backpropagation to drive the intermediate weights $\mathcal{F}(\mathbf{x})$ toward **zero** than to learn an identity transformation from scratch with non-zero weights.

### 2. Identity Shortcuts vs. Projection Shortcuts

To add the input $\mathbf{x}$ to the output $\mathcal{F}(\mathbf{x})$, dimensions must match:

- **Identity Shortcut:** Used when spatial dimensions and channel depths are identical. The input vector $\mathbf{x}$ is added directly without any extra parameters or computation.
    
- **Projection Shortcut:** Used when spatial dimensions halve or channel counts double (due to Stride 2). A $1 \times 1$ convolution with Stride 2 is applied to $\mathbf{x}$ to match the output tensor shape before addition.
    

### 3. Gradient Highway against Vanishing Gradients

During backpropagation, the derivative of $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ with respect to $\mathbf{x}$ is:

$$\frac{\partial \mathcal{H}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + 1$$

The $+1$ term acts as a **gradient highway**. Even if the intermediate weight layer gradients ($\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$) vanish toward zero in very deep networks, the gradient can still flow back unimpeded through the direct identity path.

### 4. Global Average Pooling (GAP) Over Heavy FC Layers

By replacing multi-layer dense classifiers (like the 4,096-neuron FC blocks in AlexNet and VGG) with Global Average Pooling, ResNet drastically reduced parameter complexity. ResNet18 contains only **~11.5M parameters** compared to VGG16's **~138M parameters**, despite delivering superior performance.