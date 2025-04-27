# Mathematical Foundations for ML Interview

## Core Mathematical Areas

### Linear Algebra Fundamentals

#### Vector Spaces and Operations
**Vector Norms:**
- L1 norm: $\|x\|_1 = \sum_{i=1}^n |x_i|$
- L2 norm: $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- Lp norm: $\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$
- L∞ norm: $\|x\|_{\infty} = \max_i |x_i|$

**Properties of vector norms:**
1. Non-negativity: $\|x\| \geq 0$ and $\|x\| = 0 \iff x = 0$
2. Scalar multiplication: $\|\alpha x\| = |\alpha| \cdot \|x\|$
3. Triangle inequality: $\|x + y\| \leq \|x\| + \|y\|$

**Vector operations:**
- Dot product: $x \cdot y = \sum_{i=1}^n x_i y_i = x^T y$
- Angle between vectors: $\cos \theta = \frac{x \cdot y}{\|x\|_2 \|y\|_2}$
- Orthogonality: $x \perp y \iff x \cdot y = 0$

#### Matrix Properties and Decompositions
**Matrix norms:**
- Frobenius norm: $\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\text{trace}(A^T A)}$
- Spectral norm: $\|A\|_2 = \sigma_{\max}(A)$ (largest singular value)

**Matrix rank:**
- $\text{rank}(A) = $ number of linearly independent rows = number of linearly independent columns
- $\text{rank}(A) = $ number of non-zero singular values
- For $A \in \mathbb{R}^{m \times n}$, $\text{rank}(A) \leq \min(m, n)$

**Eigendecomposition:**
- For square matrix $A \in \mathbb{R}^{n \times n}$
- $A = Q \Lambda Q^{-1}$ where $Q$ has eigenvectors as columns and $\Lambda$ is diagonal with eigenvalues
- Valid when $A$ has $n$ linearly independent eigenvectors
- For symmetric matrices: $Q$ is orthogonal, so $A = Q \Lambda Q^T$

**Singular Value Decomposition (SVD):**
- For any matrix $A \in \mathbb{R}^{m \times n}$
- $A = U \Sigma V^T$ where:
  - $U \in \mathbb{R}^{m \times m}$ is orthogonal with left singular vectors
  - $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq ... \geq 0$
  - $V \in \mathbb{R}^{n \times n}$ is orthogonal with right singular vectors
- Low-rank approximation: $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ is the best rank-k approximation to $A$

**Trace and Determinant:**
- Trace: $\text{trace}(A) = \sum_{i=1}^n a_{ii} = \sum_{i=1}^n \lambda_i$ (sum of eigenvalues)
- Determinant: $\det(A) = \prod_{i=1}^n \lambda_i$ (product of eigenvalues)
- $\det(AB) = \det(A) \det(B)$ and $\text{trace}(A+B) = \text{trace}(A) + \text{trace}(B)$

#### Key Linear Algebra Proofs

**Proof: Gradient of quadratic form $f(x) = x^T A x$**
```
Let f(x) = x^T A x where A is symmetric.

f(x + h) = (x + h)^T A (x + h)
         = x^T A x + x^T A h + h^T A x + h^T A h
         = f(x) + x^T A h + h^T A x + O(||h||^2)
         = f(x) + x^T A h + (x^T A h)^T + O(||h||^2)
         = f(x) + x^T A h + h^T A^T x + O(||h||^2)
         = f(x) + x^T A h + h^T A x + O(||h||^2)
         = f(x) + 2x^T A h + O(||h||^2)

Comparing with the definition of the gradient:
f(x + h) ≈ f(x) + ∇f(x)^T h

We get: ∇f(x) = 2Ax
```

**Proof: Matrix inversion via adjugate**
```
For a square matrix A, the adjugate adj(A) is the transpose of the cofactor matrix.
Let's prove A⁻¹ = adj(A)/det(A) for invertible matrices.

Let C be the cofactor matrix of A, so adj(A) = C^T.
For any i,j:
(AC)ij = ∑k Aik·Ckj

When i=j, this is the Laplace expansion of det(A) along row i.
When i≠j, this creates a matrix with two identical rows, so determinant = 0.

Therefore, AC = det(A)·I, or A·(C^T)^T = det(A)·I
This gives A·adj(A) = det(A)·I

Since A is invertible, det(A)≠0, so:
A⁻¹ = adj(A)/det(A)
```

### Probability and Statistics

#### Random Variables and Distributions
**Expected value and variance:**
- $\mathbb{E}[X] = \sum_x x P(X=x)$ or $\mathbb{E}[X] = \int_x x f(x) dx$
- $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
- $\text{Cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$

**Key properties:**
- Linearity of expectation: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- Variance of linear combination: $\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y) + 2ab\text{Cov}(X,Y)$
- For independent X,Y: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ and $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$

**Common distributions and their properties:**

1. **Bernoulli(p)**
   - PMF: $P(X=1) = p, P(X=0) = 1-p$
   - Mean: $\mathbb{E}[X] = p$
   - Variance: $\text{Var}(X) = p(1-p)$

2. **Binomial(n,p)**
   - PMF: $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$
   - Mean: $\mathbb{E}[X] = np$
   - Variance: $\text{Var}(X) = np(1-p)$

3. **Normal($\mu$,$\sigma^2$)**
   - PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
   - Mean: $\mathbb{E}[X] = \mu$
   - Variance: $\text{Var}(X) = \sigma^2$
   - Standard Normal: $Z \sim \mathcal{N}(0,1)$
   - Linear transformation: If $X \sim \mathcal{N}(\mu,\sigma^2)$, then $aX + b \sim \mathcal{N}(a\mu+b, a^2\sigma^2)$

4. **Poisson($\lambda$)**
   - PMF: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$
   - Mean: $\mathbb{E}[X] = \lambda$
   - Variance: $\text{Var}(X) = \lambda$

5. **Uniform(a,b)**
   - PDF: $f(x) = \frac{1}{b-a}$ for $a \leq x \leq b$
   - Mean: $\mathbb{E}[X] = \frac{a+b}{2}$
   - Variance: $\text{Var}(X) = \frac{(b-a)^2}{12}$

#### Law of Large Numbers and Central Limit Theorem
**Law of Large Numbers:**
- If $X_1, X_2, ..., X_n$ are i.i.d. with $\mathbb{E}[X_i] = \mu$, then
- $\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu$ as $n \rightarrow \infty$

**Central Limit Theorem:**
- If $X_1, X_2, ..., X_n$ are i.i.d. with $\mathbb{E}[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$, then
- $\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$ as $n \rightarrow \infty$
- Equivalently: $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$

**Proof sketch of CLT:**
```
The key insight uses characteristic functions (Fourier transforms of distributions).
Let φX(t) = E[e^{itX}] be the characteristic function of X.

1. For sum Sn = X₁ + ... + Xn of i.i.d. variables, φSn(t) = [φX(t)]^n
2. For standardized sum Zn = (Sn-nμ)/(σ√n), φZn(t) = [φX(t/σ√n)·e^{-itμ/σ√n}]^n
3. Taylor expand φX(t) = 1 + itμ - t²σ²/2 + o(t²)
4. Substitute and simplify: 
   φZn(t) → exp(-t²/2) as n→∞
5. Since exp(-t²/2) is the characteristic function of N(0,1),
   Zn converges in distribution to N(0,1)
```

#### Maximum Likelihood Estimation
**Definition:** MLE finds parameters θ that maximize the likelihood of observing the data X.
- $\hat{\theta}_{MLE} = \arg\max_{\theta} P(X|\theta) = \arg\max_{\theta} \mathcal{L}(\theta|X)$
- Often more convenient to maximize log-likelihood: $\hat{\theta}_{MLE} = \arg\max_{\theta} \log \mathcal{L}(\theta|X)$

**MLE for Gaussian distribution:**
```
Let X = {x₁, x₂, ..., xₙ} be i.i.d. samples from N(μ,σ²).

The likelihood function is:
L(μ,σ²|X) = ∏ᵢ₌₁ⁿ (1/√(2πσ²)) * exp(-(xᵢ-μ)²/(2σ²))

Taking the log:
log L(μ,σ²|X) = -n/2 * log(2π) - n/2 * log(σ²) - ∑ᵢ₌₁ⁿ(xᵢ-μ)²/(2σ²)

Taking partial derivatives and setting to zero:
∂/∂μ[log L] = ∑ᵢ₌₁ⁿ(xᵢ-μ)/σ² = 0
∂/∂σ²[log L] = -n/(2σ²) + ∑ᵢ₌₁ⁿ(xᵢ-μ)²/(2σ⁴) = 0

Solving these equations:
μ̂ = (1/n)∑ᵢ₌₁ⁿxᵢ = x̄
σ̂² = (1/n)∑ᵢ₌₁ⁿ(xᵢ-μ̂)²
```

**Properties of MLE:**
- Consistent: converges to true parameter as sample size increases
- Asymptotically normal: distribution of MLE approaches normal as sample size increases
- Asymptotically efficient: achieves Cramér-Rao lower bound asymptotically

### Calculus and Optimization

#### Derivatives and Gradients
**Chain rule for scalar functions:**
- $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$

**Gradient of a scalar function:**
- $\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right]^T$

**Directional derivative:**
- Along unit vector u: $\nabla_u f(x) = \nabla f(x) \cdot u$
- Steepest ascent direction is $\nabla f(x)$
- Steepest descent direction is $-\nabla f(x)$

**Jacobian matrix:**
- For vector function $F: \mathbb{R}^n \rightarrow \mathbb{R}^m$
- $J_F(x) = \begin{bmatrix} \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n} \end{bmatrix}$

**Hessian matrix:**
- Second derivatives of scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$
- $H_f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$
- If all partial derivatives are continuous, Hessian is symmetric

**Common derivatives for ML:**
- $\frac{d}{dx}\log(x) = \frac{1}{x}$
- $\frac{d}{dx}e^x = e^x$
- $\frac{d}{dx}\text{sigmoid}(x) = \text{sigmoid}(x)(1-\text{sigmoid}(x))$
- $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$
- $\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$

#### Optimization Techniques
**Gradient Descent:**
- Update rule: $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$
- Convergence rate for convex, Lipschitz functions: $O(1/t)$
- Convergence rate for strongly convex functions: $O(e^{-ct})$

**Newton's Method:**
- Update rule: $\theta_{t+1} = \theta_t - [H_f(\theta_t)]^{-1} \nabla f(\theta_t)$
- Convergence rate: quadratic (much faster than gradient descent)
- Drawbacks: Computing and inverting Hessian is expensive

**Stochastic Gradient Descent:**
- Update rule: $\theta_{t+1} = \theta_t - \eta \nabla f_i(\theta_t)$
- Use random sample or mini-batch at each iteration
- Convergence rate for convex functions: $O(1/\sqrt{t})$

**Momentum:**
- Update rule: 
  - $v_{t+1} = \gamma v_t + \eta \nabla f(\theta_t)$
  - $\theta_{t+1} = \theta_t - v_{t+1}$
- Helps escape shallow local minima and accelerates convergence

**Constrained optimization:**
- Lagrangian: $L(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x)$
- Karush-Kuhn-Tucker (KKT) conditions for constrained optimization problems

#### Convexity and Optimality
**Convex function:**
- $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ for all $x,y$ and $\lambda \in [0,1]$
- First-order condition: $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ for all $x,y$
- Second-order condition: Hessian is positive semi-definite everywhere

**Convex optimization properties:**
- Local minima are global minima
- First-order optimality condition: $\nabla f(x^*) = 0$
- Second-order sufficient condition: Hessian is positive definite at $x^*$

**Proof: First-order condition for convexity**
```
To prove: f is convex if and only if f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) for all x,y.

Forward direction: 
Assume f is convex. For any λ ∈ (0,1), by definition:
f(x + λ(y-x)) ≤ (1-λ)f(x) + λf(y)

Rearranging:
[f(x + λ(y-x)) - f(x)]/λ ≤ f(y) - f(x)

As λ → 0⁺, the left side → ∇f(x)ᵀ(y-x)
Therefore: ∇f(x)ᵀ(y-x) ≤ f(y) - f(x)
Rearranging: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)

Reverse direction:
Assume f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) for all x,y.
For any λ ∈ [0,1], let z = λx + (1-λ)y.

Apply the condition at point z with point x:
f(x) ≥ f(z) + ∇f(z)ᵀ(x-z)

Apply the condition at point z with point y:
f(y) ≥ f(z) + ∇f(z)ᵀ(y-z)

Multiply first inequality by λ, second by (1-λ), and add:
λf(x) + (1-λ)f(y) ≥ f(z) + ∇f(z)ᵀ[λ(x-z) + (1-λ)(y-z)]

The term in brackets equals zero since z = λx + (1-λ)y.
Therefore: λf(x) + (1-λ)f(y) ≥ f(z) = f(λx + (1-λ)y)

This establishes convexity of f.
```

### Information Theory

#### Entropy and Mutual Information
**Entropy:**
- $H(X) = -\sum_x P(X=x) \log P(X=x) = -\mathbb{E}[\log P(X)]$
- Measures uncertainty in a random variable
- Maximum entropy occurs with uniform distribution

**Conditional entropy:**
- $H(Y|X) = -\sum_{x,y} P(X=x, Y=y) \log P(Y=y|X=x) = -\mathbb{E}[\log P(Y|X)]$
- Measures remaining uncertainty in Y after observing X

**Mutual information:**
- $I(X;Y) = \sum_{x,y} P(X=x,Y=y) \log \frac{P(X=x,Y=y)}{P(X=x)P(Y=y)} = \mathbb{E}\left[\log \frac{P(X,Y)}{P(X)P(Y)}\right]$
- $I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$
- Measures information shared between X and Y

**KL Divergence:**
- $D_{KL}(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$
- Measures how different Q is from P
- Always non-negative: $D_{KL}(P||Q) \geq 0$, with equality iff P = Q
- Not symmetric: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

**Cross-entropy:**
- $H(P,Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{KL}(P||Q)$
- Used in classification loss functions

## ML Theory Derivations

### Linear Regression Derivation

**Ordinary Least Squares:**
1. **Problem formulation:**
   - Given data $\{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d, y_i \in \mathbb{R}$
   - Find $w \in \mathbb{R}^d$ that minimizes $J(w) = \sum_{i=1}^n (y_i - w^T x_i)^2$

2. **Solution derivation:**
```
In matrix form, with X = [x₁ᵀ; x₂ᵀ; ...; xₙᵀ] and y = [y₁; y₂; ...; yₙ]:
J(w) = ||y - Xw||² = (y - Xw)ᵀ(y - Xw)
     = yᵀy - yᵀXw - wᵀXᵀy + wᵀXᵀXw
     = yᵀy - 2wᵀXᵀy + wᵀXᵀXw

Taking the gradient with respect to w:
∇ₘJ(w) = -2Xᵀy + 2XᵀXw

Setting to zero:
XᵀXw = Xᵀy

If XᵀX is invertible, the unique solution is:
w* = (XᵀX)⁻¹Xᵀy

This is the normal equation for ordinary least squares.
```

3. **Properties:**
   - If $X^TX$ is invertible, the solution is unique
   - Consistent estimator: $\hat{w} \rightarrow w^*$ as $n \rightarrow \infty$
   - Minimum variance among unbiased linear estimators (Gauss-Markov theorem)

**Ridge Regression (L2 regularization):**
1. **Problem formulation:**
   - Minimize $J(w) = \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \|w\|_2^2$

2. **Solution derivation:**
```
In matrix form:
J(w) = ||y - Xw||² + λ||w||²
     = (y - Xw)ᵀ(y - Xw) + λwᵀw
     = yᵀy - 2wᵀXᵀy + wᵀXᵀXw + λwᵀw
     = yᵀy - 2wᵀXᵀy + wᵀ(XᵀX + λI)w

Taking the gradient:
∇ₘJ(w) = -2Xᵀy + 2(XᵀX + λI)w

Setting to zero:
(XᵀX + λI)w = Xᵀy

The solution is:
w* = (XᵀX + λI)⁻¹Xᵀy
```

3. **Properties:**
   - Solution always exists even if $X^TX$ is not invertible
   - Biased estimator but potentially lower variance
   - Effective for multicollinearity problems
   - Shrinks coefficients toward zero (but not to zero)

**Lasso Regression (L1 regularization):**
1. **Problem formulation:**
   - Minimize $J(w) = \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \|w\|_1$

2. **No closed-form solution**
   - Typically solved using coordinate descent or proximal gradient methods
   - Subgradient at w_j = 0 is [-1, 1], making direct calculus approach difficult

3. **Properties:**
   - Promotes sparsity: some coefficients become exactly zero
   - Feature selection built into the model
   - Can be more robust to outliers than Ridge regression

### Logistic Regression Derivation

1. **Model formulation:**
   - $P(y=1|x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$
   - $P(y=0|x) = 1 - \sigma(w^T x)$

2. **Likelihood function:**
```
L(w) = ∏ᵢ₌₁ⁿ P(y=yᵢ|xᵢ,w)
     = ∏ᵢ₌₁ⁿ σ(wᵀxᵢ)^yᵢ · (1-σ(wᵀxᵢ))^(1-yᵢ)

Taking the log-likelihood:
ℓ(w) = ∑ᵢ₌₁ⁿ [yᵢlog(σ(wᵀxᵢ)) + (1-yᵢ)log(1-σ(wᵀxᵢ))]

Using the derivative of sigmoid: σ'(z) = σ(z)(1-σ(z))
The gradient is:
∇ₘℓ(w) = ∑ᵢ₌₁ⁿ xᵢ(yᵢ - σ(wᵀxᵢ))

Since there's no closed-form solution, we use iterative methods:
w^(t+1) = w^(t) + η∑ᵢ₌₁ⁿ xᵢ(yᵢ - σ(w^(t)ᵀxᵢ))
```

3. **Properties:**
   - Maximum likelihood estimator
   - Decision boundary is linear: $w^T x = 0$
   - Output can be interpreted as probability

### Support Vector Machines Derivation

1. **Hard-margin SVM formulation:**
   - Find hyperplane $w^T x + b = 0$ that maximizes margin
   - Constraints: $y_i(w^T x_i + b) \geq 1$ for all i (points correctly classified with margin)
   - Objective: Minimize $\|w\|_2$ to maximize margin (margin width = $\frac{2}{\|w\|_2}$)

2. **Lagrangian formulation:**
```
Primal problem:
min_{w,b} 1/2 ||w||² subject to yᵢ(wᵀxᵢ + b) ≥ 1 for all i

Lagrangian:
L(w,b,α) = 1/2||w||² - ∑ᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]

Setting derivatives to zero:
∂L/∂w = w - ∑ᵢ αᵢyᵢxᵢ = 0 → w = ∑ᵢ αᵢyᵢxᵢ
∂L/∂b = -∑ᵢ αᵢyᵢ = 0 → ∑ᵢ αᵢyᵢ = 0

Substituting back:
L(α) = ∑ᵢ αᵢ - 1/2 ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ

Dual problem:
max_α ∑ᵢ αᵢ - 1/2 ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
subject to αᵢ ≥ 0 and ∑ᵢ αᵢyᵢ = 0
```

3. **Soft-margin SVM (with slack variables):**
   - Introduce slack variables $\xi_i \geq 0$ to allow for misclassifications
   - Constraints become: $y_i(w^T x_i + b) \geq 1 - \xi_i$ and $\xi_i \geq 0$
   - Objective: Minimize $\frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^n \xi_i$
   - Parameter C controls the trade-off between margin width and classification error

4. **Kernel trick:**
   - Replace dot products $x_i^T x_j$ with kernel function $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$
   - Allows implicit mapping to higher-dimensional space without computing $\phi(x)$ explicitly
   - Common kernels: 
     - Linear: $K(x_i,x_j) = x_i^T x_j$
     - Polynomial: $K(x_i,x_j) = (x_i^T x_j + c)^d$
     - RBF: $K(x_i,x_j) = \exp(-\gamma \|x_i - x_j\|^2)$

### Principal Component Analysis (PCA) Derivation

1. **Problem formulation:**
   - Find orthogonal directions that capture maximum variance in data
   - Project data onto lower-dimensional subspace while preserving most information

2. **Derivation via variance maximization:**
```
Given data matrix X = [x₁, x₂, ..., xₙ]ᵀ (centered), find unit vector u₁ such that:
u₁ = argmax_u ||Xu||² subject to ||u|| = 1

The projection variance is:
Var(Xu) = (1/n)||Xu||² = (1/n)uᵀXᵀXu = uᵀΣu

where Σ = (1/n)XᵀX is the covariance matrix.

Using Lagrange multipliers with constraint ||u|| = 1:
L(u,λ) = uᵀΣu - λ(uᵀu - 1)

Setting ∂L/∂u = 0:
Σu = λu

This is an eigenvalue problem! The solution u₁ is the eigenvector of Σ 
corresponding to the largest eigenvalue λ₁.

For subsequent components, add orthogonality constraints:
uᵢᵀuⱼ = 0 for j < i

The solution gives principal components as eigenvectors of Σ,
ordered by decreasing eigenvalues.
```

3. **Properties of PCA:**
   - Principal components are orthogonal
   - First k components give the best k-dimensional linear approximation of data
   - Eigenvalues represent variance explained by each component
   - Total variance preserved: $\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$

### Naive Bayes Derivation

1. **Model formulation:**
   - Using Bayes' rule: $P(y|x) = \frac{P(x|y)P(y)}{P(x)}$
   - "Naive" assumption: Features are conditionally independent given the class
   - This means: $P(x|y) = \prod_{j=1}^d P(x_j|y)$

2. **Classification rule:**
```
For class y ∈ {1,2,...,K}:
y* = argmax_y P(y|x) = argmax_y P(x|y)P(y)/P(x)
    = argmax_y P(x|y)P(y) 
    = argmax_y P(y)∏ⱼ₌₁ᵈ P(xⱼ|y)

Taking the log:
y* = argmax_y [log P(y) + ∑ⱼ₌₁ᵈ log P(xⱼ|y)]
```

3. **Parameter estimation:**
   - For discrete features: $P(x_j = v | y = c) = \frac{\text{count}(x_j = v, y = c)}{\text{count}(y = c)}$
   - For continuous features: Often assume Gaussian distribution
   - Prior probabilities: $P(y = c) = \frac{\text{count}(y = c)}{n}$

4. **Handling zero probabilities (smoothing):**
   - Laplace (add-one) smoothing: $P(x_j = v | y = c) = \frac{\text{count}(x_j = v, y = c) + 1}{\text{count}(y = c) + |V_j|}$
   - Where $|V_j|$ is the number of possible values for feature $j$

### Neural Network Basics

1. **Feed-forward computation:**
   - Layer l computation: $a^{(l)} = \sigma(z^{(l)}) = \sigma(W^{(l)}a^{(l-1)} + b^{(l)})$
   - Input layer: $a^{(0)} = x$
   - Output layer: $\hat{y} = a^{(L)}$

2. **Backpropagation derivation:**
```
Using MSE loss: L = (1/2)||y - a^(L)||²

Define δ^(l) = ∂L/∂z^(l) (error signal at layer l)

For output layer:
δ^(L) = -(y - a^(L)) ⊙ σ'(z^(L))

For hidden layers (using chain rule):
δ^(l) = ((W^(l+1))ᵀδ^(l+1)) ⊙ σ'(z^(l))

Parameter gradients:
∂L/∂W^(l) = δ^(l)(a^(l-1))ᵀ
∂L/∂b^(l) = δ^(l)

Update rule:
W^(l) ← W^(l) - η·∂L/∂W^(l)
b^(l) ← b^(l) - η·∂L/∂b^(l)
```

3. **Common activation functions:**
   - Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$
   - Tanh: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, $\tanh'(z) = 1 - \tanh^2(z)$
   - ReLU: $\text{ReLU}(z) = \max(0, z)$, $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$
   - Leaky ReLU: $\text{LeakyReLU}(z) = \max(\alpha z, z)$ with small $\alpha$ (e.g., 0.01)

4. **Initialization strategies:**
   - Xavier/Glorot: $W \sim \text{Uniform}(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}})$
   - He initialization: $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$

## Statistical Learning Theory

### Bias-Variance Decomposition

**Derivation for squared error:**
```
For true function f, model h, and noise ε with E[ε] = 0, Var(ε) = σ²:
y = f(x) + ε

Expected test error at x:
E[(y - h(x))²] = E[(f(x) + ε - h(x))²]
                = E[(f(x) - h(x))² + ε² + 2ε(f(x) - h(x))]
                = E[(f(x) - h(x))²] + E[ε²] + 2E[ε]E[f(x) - h(x)]
                = E[(f(x) - h(x))²] + σ²

Let h̄(x) = E[h(x)] (expected prediction over different training sets).
Then:
E[(f(x) - h(x))²] = E[(f(x) - h̄(x) + h̄(x) - h(x))²]
                   = (f(x) - h̄(x))² + E[(h̄(x) - h(x))²]
                   = Bias²(h(x)) + Var(h(x))

Therefore:
E[(y - h(x))²] = Bias²(h(x)) + Var(h(x)) + σ²
```

**Interpretation:**
- Bias: systematic error due to model's limited expressiveness
- Variance: sensitivity to training data fluctuations
- Noise: irreducible error in the data
- Bias-variance tradeoff: increasing model complexity typically decreases bias but increases variance

### Generalization and Overfitting

**Empirical Risk Minimization:**
- Training error (empirical risk): $R_{emp}(h) = \frac{1}{n}\sum_{i=1}^n L(h(x_i), y_i)$
- True risk: $R(h) = \mathbb{E}_{(x,y)}[L(h(x), y)]$

**VC Dimension and Bounds:**
- VC dimension: Maximum number of points that can be shattered (classified in all $2^n$ possible ways)
- Vapnik-Chervonenkis bound: With probability $1-\delta$,
  $R(h) \leq R_{emp}(h) + \sqrt{\frac{d(\log(2n/d) + 1) + \log(4/\delta)}{n}}$
  where $d$ is the VC dimension

**Regularization View:**
- Add regularization term to objective: $J(h) = R_{emp}(h) + \lambda \Omega(h)$
- $\Omega(h)$ controls model complexity
- $\lambda$ balances fitting data vs. keeping model simple

### Learning Curves

**Typical behavior:**
- Training error increases with more data (more constraints)
- Test error decreases with more data (better generalization)
- Both converge to irreducible error (Bayes error) as $n \rightarrow \infty$

**Diagnosing model problems:**
- High bias: Both training and test error are high and converge quickly
- High variance: Large gap between training and test error, slow convergence
- Optimal model: Small gap, both errors close to Bayes error

## Common Machine Learning Algorithms

### Decision Trees

**Information Gain:**
- Entropy: $H(S) = -\sum_{c \in C} p(c) \log_2 p(c)$
- Information gain: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$
- Gain ratio: $GR(S, A) = \frac{IG(S, A)}{H_A(S)}$ where $H_A(S) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$

**Gini Index:**
- $Gini(S) = 1 - \sum_{c \in C} p(c)^2$
- Used in CART algorithm for splitting

**Pruning strategies:**
- Pre-pruning: Stop growing tree when information gain is below threshold
- Post-pruning: Grow full tree, then remove nodes that don't improve validation performance

### Ensemble Methods

**Random Forest:**
- Bootstrap samples of data
- Subset of features at each split
- Majority voting for classification
- Reduces variance compared to single trees
- Out-of-bag error: Estimate using samples not in bootstrap

**Boosting:**
- AdaBoost: 
  - Reweight misclassified examples in each iteration
  - Final prediction: $H(x) = \text{sign}(\sum_{t=1}^T \alpha_t h_t(x))$ where $\alpha_t = \frac{1}{2} \ln(\frac{1-\epsilon_t}{\epsilon_t})$
- Gradient Boosting:
  - Sequentially fit new model to residuals of previous ensemble
  - $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$ where $h_m$ minimizes $\sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h_m(x_i))$

**Stacking:**
- Train multiple diverse base models
- Use their predictions as features for a meta-model
- Can combine strengths of different algorithms

### Advanced Topics

**Attention Mechanisms:**
- Self-attention: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- Multi-head attention: Split queries, keys, values and compute attention in parallel
- Allows learning long-range dependencies efficiently

**Generative Models:**
- GAN objective: $\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$
- Variational autoencoder objective: $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$
- Diffusion models: Progressive noising and denoising process

**Reinforcement Learning:**
- Value function: $V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s]$
- Q-function: $Q^\pi(s, a) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s, A_0 = a]$
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)]$