# ML Breadth Round Preparation

## Format
- 60-minute interview focused on broad ML knowledge
- Emphasis on mathematical foundations and practical applications
- Questions range from basic statistics to complex ML algorithms
- No coding expected, but ability to sketch pseudocode is valuable

## Core Topics to Master

### Probability & Statistics (High Likelihood)

**Q1: Explain maximum likelihood estimation and derive the MLE for Gaussian distribution.**
```
A1: Maximum likelihood estimation finds parameters θ that maximize the likelihood of observing data X:

θ_MLE = argmax_θ P(X|θ)

For computational convenience, we maximize log-likelihood:
θ_MLE = argmax_θ log(P(X|θ))

For a Gaussian distribution with parameters μ and σ²:
P(x|μ,σ²) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))

For n i.i.d. samples:
log(P(X|μ,σ²)) = -n/2·log(2π) - n/2·log(σ²) - Σ(x_i-μ)²/(2σ²)

Taking derivatives and setting to zero:
∂/∂μ[log(P(X|μ,σ²))] = Σ(x_i-μ)/σ² = 0
∂/∂σ²[log(P(X|μ,σ²))] = -n/(2σ²) + Σ(x_i-μ)²/(2σ⁴) = 0

Solving:
μ_MLE = (1/n)·Σx_i = x̄ (sample mean)
σ²_MLE = (1/n)·Σ(x_i-μ_MLE)² (sample variance)

Note that σ²_MLE is biased; the unbiased estimator uses (n-1) denominator.
```

**Q2: Explain the difference between Type I and Type II errors, and how they relate to precision and recall.**
```
A2: In hypothesis testing:
- Type I error (false positive): Rejecting null hypothesis when it's true
- Type II error (false negative): Failing to reject null hypothesis when it's false

For a classification problem:
- Type I error rate = FP/(FP+TN) = 1-specificity
- Type II error rate = FN/(TP+FN) = 1-sensitivity

Relation to precision/recall:
- Precision = TP/(TP+FP): Inversely related to Type I errors
- Recall = TP/(TP+FN) = Sensitivity: Inversely related to Type II errors

In practical terms:
- Increasing threshold reduces Type I errors but increases Type II errors
- Decreasing threshold reduces Type II errors but increases Type I errors

This creates a trade-off visualized by the precision-recall curve, with F1-score representing their harmonic mean:
F1 = 2·(precision·recall)/(precision+recall)
```

**Q3: Derive the Bayes error rate and explain its significance.**
```
A3: The Bayes error rate is the theoretical minimum error achievable by any classifier.

For a binary classification problem with features X and class Y:
P(error) = ∫ min(P(Y=0|X=x), P(Y=1|X=x))·p(x)dx

This represents the error from optimal decision rule: choose class with highest posterior probability.

Deriving from Bayes' theorem:
P(Y=k|X=x) = [P(X=x|Y=k)·P(Y=k)]/P(X=x)

The Bayes error rate quantifies the fundamental uncertainty in the problem due to:
1. Class overlap in feature space
2. Insufficient discriminative information in features

Significance:
- Provides theoretical lower bound on achievable error
- Benchmark for model performance (error - Bayes error = reducible error)
- Helps determine if additional features are needed

For Gaussian distributions with equal covariance matrices, Bayes error can be computed using the Mahalanobis distance between class means.
```

### Classical ML Algorithms (Very High Likelihood)

**Q1: Compare and contrast different ensemble methods (bagging, boosting, stacking).**
```
A1: Ensemble methods combine multiple models to improve performance:

Bagging (Bootstrap Aggregating):
- Trains models independently on bootstrap samples
- Reduces variance without affecting bias
- Mathematical formulation: f(x) = (1/M)·∑ᵢf_i(x)
- Example: Random Forest uses bagging with decision trees
- Computational advantage: Parallelizable training

Boosting:
- Trains models sequentially, focusing on previous errors
- Reduces bias and can reduce variance
- AdaBoost formula: f(x) = ∑ᵢα_i·h_i(x) where α_i = 0.5·ln((1-ε_i)/ε_i)
- Gradient Boosting: f_m(x) = f_{m-1}(x) - η·∇L(f_{m-1}(x))
- Tends to overfit with noisy data

Stacking:
- Uses predictions of base models as features for meta-model
- Mathematical formulation: f(x) = g(f₁(x), f₂(x), ..., f_k(x))
- Learns optimal combination of base models
- More flexible than weighted averaging
- Requires careful cross-validation to prevent leakage

Key differences:
- Variance reduction: Bagging > Stacking > Boosting
- Bias reduction: Boosting > Stacking > Bagging
- Computational cost: Boosting > Stacking > Bagging
- Sensitivity to noise: Boosting > Stacking > Bagging
```

**Q2: Explain the mathematical foundations of SVM and derive the optimization problem.**
```
A2: Support Vector Machines find a hyperplane that maximizes the margin between classes.

For linearly separable data:
- Hyperplane equation: w^T·x + b = 0
- Decision function: f(x) = sign(w^T·x + b)
- Constraint: y_i(w^T·x_i + b) ≥ 1 for all i

The margin width equals 2/||w||, so maximizing margin means minimizing ||w||.
This gives optimization problem:
min 0.5·||w||² subject to y_i(w^T·x_i + b) ≥ 1 for all i

Using Lagrangian multipliers:
L(w,b,α) = 0.5·||w||² - ∑ᵢα_i[y_i(w^T·x_i + b) - 1]

Taking derivatives and setting to zero:
∂L/∂w = w - ∑ᵢα_i·y_i·x_i = 0 → w = ∑ᵢα_i·y_i·x_i
∂L/∂b = -∑ᵢα_i·y_i = 0 → ∑ᵢα_i·y_i = 0

Substituting back yields dual form:
max ∑ᵢα_i - 0.5·∑ᵢ∑ⱼα_i·α_j·y_i·y_j·(x_i^T·x_j)
subject to α_i ≥ 0 and ∑ᵢα_i·y_i = 0

For non-separable data, introduce slack variables ξ_i ≥ 0:
min 0.5·||w||² + C·∑ᵢξ_i
subject to y_i(w^T·x_i + b) ≥ 1 - ξ_i for all i

The kernel trick extends SVMs to nonlinear boundaries by replacing dot products:
K(x_i,x_j) = φ(x_i)^T·φ(x_j)
where φ transforms features to higher-dimensional space.
```

**Q3: Explain Principal Component Analysis and derive the optimization problem.**
```
A3: PCA finds orthogonal directions of maximum variance in the data.

For data matrix X (n×d), PCA projects data onto k principal components:

Mathematically, we seek vectors w that maximize:
variance(w^T·X) = w^T·Σ·w

Where Σ = (1/n)·X^T·X is the covariance matrix, subject to ||w||=1.

Using Lagrange multipliers:
L(w,λ) = w^T·Σ·w - λ(w^T·w - 1)

Setting ∂L/∂w = 0:
Σ·w = λ·w

This is an eigenvalue problem! Principal components are eigenvectors of Σ, and corresponding eigenvalues represent variance along each component.

Algorithm:
1. Center data: X' = X - μ_X
2. Compute covariance matrix: Σ = (1/n)·X'^T·X'
3. Find eigenvectors/eigenvalues: Σ·w_i = λ_i·w_i
4. Sort eigenvectors by decreasing eigenvalues
5. Project data: Z = X'·W where W = [w₁, w₂, ..., w_k]

Properties:
- First k components capture maximum possible variance
- Components are orthogonal: w_i^T·w_j = 0 for i≠j
- Reconstruction error = ∑ᵢ₌ₖ₊₁ᵈλ_i (sum of unused eigenvalues)
- Dimensionality selection: Choose k to preserve X% of variance
```

### Deep Learning Algorithms (Very High Likelihood)

**Q1: Explain different optimization algorithms for neural networks.**
```
A1: Neural network optimization algorithms improve upon vanilla gradient descent:

1. Stochastic Gradient Descent (SGD):
   w ← w - η·∇L_i(w)
   - Uses single sample gradient
   - High variance, but computationally efficient
   - Often takes many iterations to converge

2. Momentum:
   v ← γ·v + η·∇L(w)
   w ← w - v
   - Accelerates convergence (γ typically 0.9)
   - Helps overcome local minima
   - Dampens oscillations in ravine-like surfaces

3. RMSprop:
   E[g²] ← β·E[g²] + (1-β)·(∇L(w))²
   w ← w - η·∇L(w)/√(E[g²] + ε)
   - Adapts learning rates per-parameter
   - Divides by moving average of squared gradients
   - Works well with non-stationary objectives

4. Adam (Adaptive Moment Estimation):
   m ← β₁·m + (1-β₁)·∇L(w)   [first moment]
   v ← β₂·v + (1-β₂)·(∇L(w))² [second moment]
   m̂ ← m/(1-β₁ᵗ)  [bias correction]
   v̂ ← v/(1-β₂ᵗ)  [bias correction]
   w ← w - η·m̂/√(v̂ + ε)
   - Combines momentum and RMSprop
   - Default choice for many applications
   - Typical values: β₁=0.9, β₂=0.999, ε=10⁻⁸

5. AdamW:
   Same as Adam but with decoupled weight decay:
   w ← w - η·m̂/√(v̂ + ε) - η·λ·w
   - Better generalization than L2 regularization
   - State-of-the-art for transformer models

Practical considerations:
- Learning rate schedules often used with all methods
- Adam converges faster but SGD+Momentum may generalize better
- Second-order methods (L-BFGS) exist but rarely used due to memory requirements
```

**Q2: Explain how Batch Normalization works and why it's effective.**
```
A2: Batch Normalization normalizes activations within a mini-batch:

For layer inputs x = {x₁...x_m} over mini-batch B:

1. Compute batch statistics:
   μ_B = (1/m)·∑ᵢx_i
   σ²_B = (1/m)·∑ᵢ(x_i - μ_B)²

2. Normalize activations:
   x̂_i = (x_i - μ_B)/√(σ²_B + ε)

3. Scale and shift (learnable parameters):
   y_i = γ·x̂_i + β

During inference, use running averages:
E[x] = E_B[μ_B]   (updated during training)
Var[x] = E_B[σ²_B]·(m/(m-1))  (bias correction)

Theoretical benefits:
1. Reduces internal covariate shift:
   - Stabilizes distribution of network activations
   - Makes optimization landscape smoother

2. Regularization effect:
   - Each sample is normalized using batch statistics
   - Adds noise proportional to 1/√batch_size

3. Allows higher learning rates:
   - Prevents exploding/vanishing gradients
   - Mathematical proof: BN bounds activation gradients

4. Reduces dependency on initialization:
   - Post-BN activations have controlled distribution
   - Enables training of very deep networks

Practical considerations:
- Ineffective for small batch sizes (use Layer Norm instead)
- For CNNs, normalize across batch and spatial dimensions
- For RNNs, specialized variants (Layer Norm) perform better
- For transformers, Layer Norm is preferred
```

**Q3: Compare architectures: CNNs vs. RNNs vs. Transformers.**
```
A3: Key architectural differences between major neural network families:

Convolutional Neural Networks (CNNs):
- Core operation: Convolution (w * x) - local receptive fields
- Mathematical property: Parameter sharing + translation equivariance
- Inductive bias: Locality, spatial hierarchy
- Computational complexity: O(k²·c·n) where k=kernel size, c=channels, n=pixels
- Memory complexity: O(c·n) with activation checkpointing
- Key innovation: Hierarchical feature extraction with parameter efficiency
- Limitations: Limited global context, fixed receptive fields

Recurrent Neural Networks (RNNs/LSTMs/GRUs):
- Core operation: h_t = f(h_{t-1}, x_t) - sequential state updates
- Mathematical property: Parameter sharing across time steps
- Inductive bias: Temporal continuity, markovian dynamics
- Computational complexity: O(d²·T) where d=hidden dim, T=sequence length
- Memory complexity: O(d·T) for BPTT, can be O(d) with gradient truncation
- Key innovation: LSTM/GRU gating mechanisms to control information flow
- Limitations: Sequential computation, gradient vanishing/exploding

Transformers:
- Core operation: Attention(Q,K,V) = softmax(QK^T/√d)·V - global interaction
- Mathematical property: Full pairwise connectivity
- Inductive bias: Very weak (position encodings needed)
- Computational complexity: O(T²·d) for self-attention
- Memory complexity: O(T²·d) from attention maps
- Key innovation: Parallel sequence processing with attention mechanism
- Limitations: Quadratic complexity, lacks intrinsic position awareness

Architectural trade-offs:
- Parameter efficiency: CNNs > RNNs > Transformers
- Long-range dependencies: Transformers > LSTMs > CNNs
- Parallelizability: Transformers ≈ CNNs > RNNs
- Inductive bias strength: CNNs > RNNs > Transformers
- Data efficiency: CNNs > RNNs > Transformers

Recent hybrid approaches:
- Vision Transformers (ViT): CNN-like patching + transformer processing
- Convolutional LSTMs: Convolution operation within recurrent cells
- Linear attention: Reduces transformer complexity to O(T·d)
```

### Evaluation Metrics and Methodologies (Medium Likelihood)

**Q1: Explain when to use different evaluation metrics for classification.**
```
A1: Classification metrics serve different purposes depending on the problem:

1. Accuracy = (TP+TN)/(TP+FP+TN+FN)
   - Use when: Classes balanced, equal misclassification costs
   - Limitations: Misleading with imbalanced data
   - Example: General product categorization

2. Precision = TP/(TP+FP)
   - Use when: False positives are costly
   - Mathematical interpretation: P(correct | predicted positive)
   - Example: Spam detection, fraud alerts

3. Recall = TP/(TP+FN)
   - Use when: False negatives are costly
   - Mathematical interpretation: P(predicted positive | actually positive)
   - Example: Disease detection, safety critical systems

4. F1-Score = 2·(precision·recall)/(precision+recall)
   - Use when: Balance between precision/recall needed
   - Mathematical interpretation: Harmonic mean of precision and recall
   - Example: Information retrieval, document classification

5. AUC-ROC = P(score(positive) > score(negative))
   - Use when: Ranking performance matters, threshold-invariant metric needed
   - Mathematical interpretation: Probability of correct ranking
   - Example: Recommendation systems, risk scoring

6. Log-loss = -(1/n)·∑(y·log(p) + (1-y)·log(1-p))
   - Use when: Probabilistic predictions matter
   - Mathematical interpretation: Cross-entropy between predictions and truth
   - Example: Calibrated risk assessments, ensemble components

7. Cohen's Kappa = (po-pe)/(1-pe) where po=observed agreement, pe=expected agreement
   - Use when: Accounting for chance agreement is important
   - Mathematical interpretation: Agreement normalized by chance
   - Example: Multi-annotator scenarios, medical diagnostics

Decision considerations:
- Class imbalance: Precision-recall over accuracy
- Misclassification costs: Weight metrics accordingly
- Deployment constraints: Consider operational thresholds
- Interpretability needs: Simpler metrics for stakeholders
```

**Q2: Discuss cross-validation strategies and when to use each.**
```
A2: Cross-validation strategies evaluate model performance with limited data:

1. k-Fold Cross Validation:
   - Algorithm: Split data into k folds, train on k-1, test on remaining
   - Mathematical formulation: Error = (1/k)·∑ᵏᵢ₌₁Error_i
   - Use when: General-purpose evaluation with sufficient data
   - Typical values: k=5,10 balances bias-variance in estimate

2. Stratified k-Fold:
   - Maintains class distribution in each fold
   - Mathematical constraint: P(Y=y|fold=i) ≈ P(Y=y) for all i,y
   - Use when: Imbalanced classification problems
   - Advantage: Reduces variance in performance estimate

3. Leave-One-Out CV:
   - Special case where k=n (sample size)
   - Mathematical equivalence: Jackknife resampling
   - Use when: Very small datasets where data is precious
   - Drawback: High computational cost, high variance

4. Time Series CV (expanding window):
   - Train: [0,t), Test: [t,t+w) with increasing t
   - Respects temporal dependencies
   - Use when: Time series forecasting, sequential data
   - Avoids data leakage from future observations

5. Group k-Fold:
   - Ensures groups (e.g., patients) don't span train/test
   - Mathematical constraint: G_train ∩ G_test = ∅ for all folds
   - Use when: Data has group structure or dependencies
   - Example: Medical data with multiple samples per patient

6. Nested CV:
   - Outer loop evaluates performance, inner loop tunes hyperparameters
   - Provides unbiased estimate of tuned model performance
   - Use when: Hyperparameter selection is part of modeling
   - Computational complexity: O(k_outer·k_inner)

Statistical considerations:
- Bias: LOOCV < k-Fold < Holdout
- Variance: LOOCV > k-Fold > Holdout
- Computational cost: LOOCV > k-Fold > Holdout
- Minimum recommended samples: 5-fold CV needs ~50 samples
```

**Q3: Explain how to handle class imbalance in ML.**
```
A3: Class imbalance strategies operate at different levels:

Data-level approaches:
1. Random Undersampling:
   - Randomly remove majority samples until balanced
   - Risk: Discards potentially valuable information
   - Mathematical effect: Changes class priors P(Y)

2. Random Oversampling:
   - Duplicate minority samples until balanced
   - Risk: Overfitting to minority examples
   - Effect: Increases weight of minority samples in loss

3. SMOTE (Synthetic Minority Oversampling TEchnique):
   - Algorithm: Generate synthetic samples by interpolating between minority neighbors
   - x_new = x_i + λ·(x_j - x_i) where λ ∈ [0,1]
   - Advantage: Creates diverse synthetic samples
   - Risk: May create unrealistic samples in feature space

4. ADASYN (Adaptive Synthetic Sampling):
   - Extends SMOTE by generating more samples for difficult cases
   - Weight generation by classification difficulty 
   - More effective for complex decision boundaries

Algorithm-level approaches:
1. Cost-sensitive Learning:
   - Modify loss function: L(y,ŷ) = w_y·L_original(y,ŷ)
   - Weight inversely proportional to frequency: w_y ∝ 1/P(Y=y)
   - Theoretically optimal for adjusting decision boundary

2. Class-weighted Ensemble:
   - Train multiple models with different sampling strategies
   - Combine with weights inversely proportional to class frequency
   - Reduces variance from any single resampling approach

3. Threshold Adjustment:
   - Calculate P(Y=1|X) as usual, adjust decision threshold
   - Optimal threshold from ROC curve for desired precision/recall
   - Cost: Does not improve ranking performance

Evaluation considerations:
- Avoid accuracy; use precision-recall AUC or F1
- Stratified cross-validation essential
- Area under precision-recall curve preferred over ROC when highly imbalanced

Amazon applications:
- Product defect detection (rare positives)
- Fraud detection in transactions (rare positives)
- Customer churn prediction (class imbalance varies)
```

## ML Breadth Success Checklist
- [ ] Explain concepts with precise mathematical notation
- [ ] Compare and contrast related algorithms with specific tradeoffs
- [ ] Provide examples relevant to Amazon applications
- [ ] Discuss practical implementation considerations
- [ ] Acknowledge computational and statistical limitations
- [ ] Demonstrate awareness of recent advancements
- [ ] Connect theoretical concepts to practical applications
- [ ] Show systematic thinking about evaluation and deployment
- [ ] Be prepared to sketch algorithms when asked