# Technical Phone Screen Preparation

## Format
- 45-60 minutes with either hiring manager or senior team member
- Focus on ML fundamentals, Python coding, and resume verification
- 2-3 technical questions + 1-2 behavioral questions
- Usually conducted via online coding platform like CoderPad

## Question Categories

### Math & Statistics (40% likelihood)

**Q1: Explain bias-variance tradeoff mathematically.**
```
A1: The expected test error can be decomposed as:

E[(y - ŷ)²] = (Bias[ŷ])² + Var[ŷ] + σ²

Where:
- Bias[ŷ] = E[ŷ] - y (systematic error)
- Var[ŷ] = E[(ŷ - E[ŷ])²] (estimation variance)
- σ² is irreducible error

In practice, as model complexity increases:
- Bias decreases (better fit to training data)
- Variance increases (more sensitivity to training samples)

The optimal model minimizes their sum. For neural networks, regularization techniques like dropout (p=0.5) reduce variance while accepting some bias.
```

**Q2: Derive the gradient descent update for logistic regression.**
```
A2: For logistic regression:
P(y=1|x) = σ(w^T x) where σ(z) = 1/(1+e^(-z))

The log-likelihood for n samples is:
L(w) = Σ[y_i log(σ(w^T x_i)) + (1-y_i)log(1-σ(w^T x_i))]

Taking derivative w.r.t w:
∇L(w) = Σ[x_i(y_i - σ(w^T x_i))]

The gradient descent update is:
w_new = w_old + α∇L(w) 
     = w_old + αΣ[x_i(y_i - σ(w^T x_i))]

For stochastic gradient descent with a single sample:
w_new = w_old + α·x_i(y_i - σ(w^T x_i))
```

**Q3: Explain L1 vs L2 regularization effects mathematically.**
```
A3: 
L1 regularization adds λ||w||₁ = λΣ|w_i| to loss function.
L2 regularization adds λ||w||₂² = λΣw_i² to loss function.

L1 effect on gradient: ∂/∂w_i = ... + λ·sign(w_i)
L2 effect on gradient: ∂/∂w_i = ... + 2λw_i

Key difference: L1 applies constant penalty regardless of weight magnitude, promoting sparsity by pushing weights exactly to zero. L2 applies penalty proportional to weight, shrinking larger weights more but rarely to exactly zero.

Mathematically, L1 is equivalent to MAP estimation with Laplace prior:
p(w) ∝ exp(-λ||w||₁)

While L2 is equivalent to MAP with Gaussian prior:
p(w) ∝ exp(-λ||w||₂²)
```

### Machine Learning Implementation (30% likelihood)

**Q1: Implement k-means clustering from scratch.**
```python
def kmeans(X, k, max_iters=100):
    # Randomly initialize k centroids
    n_samples, n_features = X.shape
    centroids_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroids_idx]
    
    for _ in range(max_iters):
        # Assign each point to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids based on assigned points
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

**Q2: Implement a function to compute precision, recall, and F1 score.**
```python
def classification_metrics(y_true, y_pred):
    # Calculate true positives, false positives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

**Q3: Implement a simple neural network forward pass.**
```python
def nn_forward(X, parameters):
    """
    Implements forward pass of 2-layer neural network
    
    Args:
        X: Input data, shape (n_features, n_examples)
        parameters: Dictionary with W1, b1, W2, b2
    
    Returns:
        A2: Output of the network
        cache: Values needed for backward pass
    """
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    
    # First layer
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    
    # Second layer
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'X': X}
    
    return A2, cache
```

### Deep Learning Concepts (20% likelihood)

**Q1: Explain vanishing/exploding gradients and how to address them.**
```
A1: Vanishing/exploding gradients occur during backpropagation in deep networks.

Mathematically, with chain rule, for an L-layer network:
∂L/∂w₁ = ∂L/∂y ⋅ ∂y/∂a_L ⋅ ∂a_L/∂z_L ⋅ ... ⋅ ∂a₂/∂z₂ ⋅ ∂z₂/∂a₁ ⋅ ∂a₁/∂z₁ ⋅ ∂z₁/∂w₁

Each ∂a_i/∂z_i term depends on activation function derivative:
- For sigmoid: σ'(z) = σ(z)(1-σ(z)) ≤ 0.25 for all z
- For tanh: tanh'(z) ≤ 1 for all z

With multiple layers (L large):
- Vanishing: Product of many terms < 1 approaches 0
- Exploding: Product of many terms > 1 approaches ∞

Solutions:
1. Activation functions: ReLU derivatives are 0 or 1, avoiding vanishing
2. Weight initialization: He init for ReLU, Xavier/Glorot for tanh 
3. Batch normalization: Standardizes activations (μ=0, σ²=1)
4. Skip connections: ResNet formula h(x) = F(x) + x bypasses problematic layers
5. LSTMs/GRUs: Gating mechanisms preserve gradients
```

**Q2: Explain how dropout works and its mathematical interpretation.**
```
A2: Dropout randomly deactivates neurons during training with probability p (typically 0.5).

For a layer with output h = [h₁, h₂, ..., h_n], we generate mask:
m = [m₁, m₂, ..., m_n] where m_i ~ Bernoulli(p)

The forward pass becomes:
h̃ = m ⊙ h / p  (element-wise multiplication with rescaling)

During inference, all neurons are active with no rescaling.

Mathematically, dropout performs:
1. Bayesian approximation: Marginalization over model uncertainty
2. Ensemble learning: Training ~2^n different thinned networks
3. Adaptive regularization: Stronger for correlated features

Convolutional layers typically use lower dropout (p=0.1-0.2) due to spatial redundancy.
```

**Q3: Explain the Transformer architecture and self-attention mechanism.**
```
A3: Transformers replace recurrence with attention mechanisms.

Self-attention computes:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Where for each token:
- Q = XW_Q: Query transformation
- K = XW_K: Key transformation 
- V = XW_V: Value transformation

For a sequence of n tokens, compute n² attention weights forming attention matrix A where:
A_ij = softmax(q_i·k_j^T/√d_k) = attention from token i to token j

The complete Transformer includes:
1. Multi-head attention: h parallel attention functions
2. Position-wise FFN: ReLU(XW₁ + b₁)W₂ + b₂
3. Residual connections + layer normalization
4. Positional encodings: sin/cos functions at different frequencies

Advantages:
- O(1) path length between any tokens vs O(n) in RNNs
- Parallelizable computation vs sequential in RNNs
- Better long-range dependency modeling
```

### Resume-Based Questions (10% likelihood)

**Q1: Explain the most complex ML algorithm you've implemented.**
```
Answer should include:
1. Algorithm chosen (with mathematical formulation)
2. Implementation challenges faced
3. Performance metrics achieved
4. Optimization techniques applied
5. Production considerations addressed
```

**Q2: Critique your current ML project methodology.**
```
Answer should include:
1. Current workflow (data processing → model selection → training → evaluation)
2. Identified bottlenecks (computational, methodological)
3. Alternative approaches considered
4. Quantifiable improvements possible
5. Lessons learned from implementation
```

## Behavioral Questions (1-2 likely)

**Q1: Tell me about a time you had to learn a new technology quickly.**
```
Structure using STAR:
- Situation: Project requiring unfamiliar deep learning architecture.
- Task: Implement feature extraction model under 2-week deadline.
- Action: Created learning plan focusing on conceptual understanding first, then code implementation, using documentation and research papers rather than tutorials.
- Result: Delivered model with 97.3% accuracy, 2 days before deadline.

Amazon Leadership Principles demonstrated:
- Learn and Be Curious
- Deliver Results
```

**Q2: How do you handle situations where there's insufficient data?**
```
Structure using STAR:
- Situation: Project classifying rare manufacturing defects.
- Task: Build model with only 50 examples of target class.
- Action: Implemented data augmentation (rotation, scaling), transfer learning from similar domain, and semi-supervised approach using unlabeled data with confidence thresholding.
- Result: Improved F1-score from 0.67 to 0.82 despite data limitations.

Amazon Leadership Principles demonstrated:
- Invent and Simplify
- Bias for Action
```

## Phone Screen Success Checklist
- [ ] Clearly verbalize thought process during problem-solving
- [ ] Ask clarifying questions before implementation
- [ ] Analyze solution time/space complexity
- [ ] Test code with example cases
- [ ] Propose optimizations or alternative approaches
- [ ] Connect answers to Amazon domain when possible
- [ ] Relate experiences to Leadership Principles
- [ ] Show curiosity about the team's specific ML projects