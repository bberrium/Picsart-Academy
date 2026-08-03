Here is a complete, step-by-step visual pipeline of how a convolutional layer processes data—from taking a 3D input tensor down to producing a full output volume.

## 1. The Setup: Matching Input Patch to 3D Kernel

A single kernel has the exact same depth as the input tensor ($C_{\text{in}}$). It overlays a specific spatial patch on the input image:

```
=========================================
 1. INPUT TENSOR: 5x5x3 (H x W x C_in)
=========================================

  Channel 3 (Blue)  ┌───┬───┬───┬───┬───┐
  Channel 2 (Green) │ * │ * │ * │   │   │
  Channel 1 (Red) ──┼───┼───┼───┤   │
                    │ * │ * │ * │   │
                    ├───┼───┼───┤   │
                    │ * │ * │ * │   │
                    └───┴───┴───┴───┘
                    ▲
                    └── 3x3 Patch across
                        ALL 3 channels

                  │
                  │ OVERLAYS / MATRICES WITH
                  ▼

=========================================
 2. SINGLE KERNEL: 3x3x3 (K_H x K_W x C_in)
=========================================

  Channel 3 Weights ┌───┬───┬───┐
  Channel 2 Weights │ w │ w │ w │
  Channel 1 Weights ┼───┼───┼───┤
                    │ w │ w │ w │
                    ├───┼───┼───┤
                    │ w │ w │ w │
                    └───┴───┴───┘
                    ▲
                    └── 3x3x3 Learnable
                        Weights
    
```

$$\text{Input Patch } (X) = \begin{bmatrix} X_{R} \\ X_{G} \\ X_{B} \end{bmatrix} = \begin{bmatrix} \begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{pmatrix} \\ \begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{pmatrix} \\ \begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{pmatrix} \end{bmatrix}$$

$$\text{Kernel Weights } (W) = \begin{bmatrix} W_{R} \\ W_{G} \\ W_{B} \end{bmatrix} = \begin{bmatrix} \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{pmatrix} \\ \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{pmatrix} \\ \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{pmatrix} \end{bmatrix}$$


$$\text{Output Scalar } y = \text{ReLU}\left( \sum (X_{R} \odot W_{R}) + \sum (X_{G} \odot W_{G}) + \sum (X_{B} \odot W_{B}) + b \right)$$
## 2. The Mechanics at a Single Location

Here is what happens inside that single patch to generate **one number**:

```
               INSPECTING A SINGLE SPATIAL POSITION (0, 0)
               ==========================================

   CHANNEL 1 (Red)         CHANNEL 2 (Green)        CHANNEL 3 (Blue)
   ┌───┬───┬───┐           ┌───┬───┬───┐            ┌───┬───┬───┐
   │ 1 │ 0 │ 2 │ Input     │ 0 │ 1 │ 1 │ Input      │ 2 │ 1 │ 0 │ Input
   ├───┼───┼───┤           ├───┼───┼───┤            ├───┼───┼───┤
   │ 0 │ 1 │ 1 │   *       │ 2 │ 0 │ 1 │   *        │ 1 │ 0 │ 1 │   *
   ├───┼───┼───┤           ├───┼───┼───┤            ├───┼───┼───┤
   │ 1 │ 0 │ 1 │           │ 1 │ 1 │ 0 │            │ 0 │ 2 │ 1 │
   └───┴───┴───┘           └───┴───┴───┘            └───┴───┴───┘
   ┌───┬───┬───┐           ┌───┬───┬───┐            ┌───┬───┬───┐
   │ 1 │-1 │ 0 │ Kernel    │ 0 │ 1 │-1 │ Kernel     │ 1 │ 0 │ 1 │ Kernel
   ├───┼───┼───┤           ├───┼───┼───┤            ├───┼───┼───┤
   │ 0 │ 2 │-1 │           │ 1 │ 0 │ 1 │            │-1 │ 1 │ 0 │
   ├───┼───┼───┤           ├───┼───┼───┤            ├───┼───┼───┤
   │ 1 │ 0 │ 1 │           │ 0 │-1 │ 1 │            │ 1 │ 0 │ 2 │
   └───┴───┴───┘           └───┴───┴───┘            └───┴───┴───┘
         │                       │                        │
         ▼                       ▼                        ▼
    (9 products)            (9 products)             (9 products)
   
   ========================================================================
   STEP 1: Multiply element-wise across all 3 channels (27 multiplications)
   
   STEP 2: Sum all 27 products together (Depth Collapse)
           Sum = (1 - 0 + 0 + 0 + 2 - 1 + 1 + 0 + 1)   [Ch 1 Sum = 4]
               + (0 + 1 - 1 + 2 + 0 + 1 + 0 - 1 + 0)   [Ch 2 Sum = 2]
               + (2 + 0 + 0 - 1 + 0 + 0 + 0 + 0 + 2)   [Ch 3 Sum = 3]
           ─────────────────────────────────────────
           Total Scalar Sum = 9.0

   STEP 3: Add Bias
           9.0 + (Bias b = -2.0) = 7.0

   STEP 4: Apply Activation Function
           ReLU(7.0) = 7.0  ──────►  [ Single Value at Output Grid (0,0) ]
```

## 3. Sliding Across Space (Building 1 Feature Map)

The single kernel shifts across the spatial dimensions using your defined **Stride**. Every step outputs one scalar into a 2D grid:

Plaintext

```
STEP A: Top-Left (0,0)      STEP B: Shift Right (0,1)    STEP C: Shift Down (1,0)
┌───┬───┬───┬───┐          ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
│ X │ X │ X │   │          │   │ X │ X │ X │            │   │   │   │   │
├───┼───┼───┼───┤          ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
│ X │ X │ X │   │          │   │ X │ X │ X │            │ X │ X │ X │   │
├───┼───┼───┼───┤          ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
│ X │ X │ X │   │          │   │ X │ X │ X │            │ X │ X │ X │   │
└───┴───┴───┴───┘          └───┴───┴───┴───┘            └───┴───┴───┴───┘
         │                          │                            │
         ▼                          ▼                            ▼
  ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐        ┌─────┬─────┬─────┐
  │ 7.0 │     │     │      │ 7.0 │ 3.5 │     │        │ 7.0 │ 3.5 │     │
  ├─────┼─────┼─────┤      ├─────┼─────┼─────┤        ├─────┼─────┼─────┤
  │     │     │     │      │     │     │     │        │ 0.0 │     │     │
  └─────┴─────┴─────┘      └─────┴─────┴─────┘        └─────┴─────┴─────┘
  
  RESULT OF 1 KERNEL SLIDING:
  ──► Produces ONE 2D Feature Map of shape (H_out x W_out)
```

## 4. Stacking Multiple Kernels (Building the Final 3D Output Tensor)

To generate multiple output channels ($C_{\text{out}}$), repeat the process using independent 3D kernels:

Plaintext

```
  Kernel 1 (3x3x3)  ──►  Slides across image  ──►  [ 2D Feature Map 1 ]
  Kernel 2 (3x3x3)  ──►  Slides across image  ──►  [ 2D Feature Map 2 ]
  Kernel 3 (3x3x3)  ──►  Slides across image  ──►  [ 2D Feature Map 3 ]
       ...                                                ...
  Kernel C_out      ──►  Slides across image  ──►  [ 2D Feature Map C_out ]


                    FINAL OUTPUT TENSOR (H_out x W_out x C_out)
                    ┌──────────────────────────┐
                   /   Feature Map C_out      /│
                  /   ...                    / │
                 /   Feature Map 2          /  │  Depth = C_out
                ┌──────────────────────────┐   │  (Number of Kernels)
                │                          │   │
                │                          │  /
                │                          │ /
                └──────────────────────────┘/
                 ◄──────── W_out ──────────►
```

## Summary Matrix Flow

$$\text{Input: } (H \times W \times C_{\text{in}})$$

$$\downarrow$$

$$\text{Apply } C_{\text{out}} \text{ Kernels of size } (K_H \times K_W \times C_{\text{in}})$$

$$\downarrow$$

$$\text{Output: } (H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}})$$