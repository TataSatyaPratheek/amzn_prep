# ML System Design Round Preparation

## Format
- 60-minute interview assessing your ability to design practical ML systems
- Focus on breaking down business problems into ML components
- Heavy emphasis on end-to-end systems, not just model architecture
- Interactive whiteboarding expected throughout

## Key Components to Address

### Problem Framing (Always Required)

**Key Elements to Include:**
1. Clarify Business Objective
   - Translate business problem to technical formulation
   - Articulate success metrics (business & technical KPIs)
   - Align ML approach with business value
   
2. Formulate as ML Problem
   - Identify appropriate ML paradigm (classification, regression, ranking, etc.)
   - Define precise mathematical formulation:
     - Input space X
     - Output space Y
     - Target function f: X → Y
     - Loss function L(y, ŷ)
   - Justify formulation with clear reasoning
   
3. Constraint Analysis
   - Latency requirements (inference time constraints)
   - Throughput needs (QPS expectations)
   - Resource limitations (memory, compute)
   - Regulatory/compliance considerations
   - Explainability requirements

### Data Strategy (Always Required)

**Key Elements to Include:**
1. Data Requirements
   - Feature identification with justification
   - Data volume estimates: "For daily user recommendation with 10M users, we need ~1TB raw interaction data"
   - Data freshness needs: "Price features require hourly updates for market responsiveness"
   - Data source mapping (internal, external, synthetic)
   
2. Feature Engineering
   - Transformations with mathematical formulas
   - Feature crossing techniques for specific domains
   - Feature selection methodology with metrics
   - Handling categorical features (embedding dimensions)
   
3. Data Pipeline Architecture
   - Batch vs streaming considerations with tradeoffs
   - Feature store design for reusability
   - Data validation checks (statistical, semantic)
   - Handling late-arriving data or corrections

### Modeling Approach (Always Required)

**Key Elements to Include:**
1. Model Selection
   - Candidate algorithms with pros/cons analysis
   - Baseline approaches (simple but robust)
   - SOTA considerations (with implementation tradeoffs)
   - Mathematical formulation of chosen models
   
2. Training Methodology
   - Loss function selection with justification
   - Optimization algorithm with hyperparameter strategy
   - Regularization approach with mathematical details
   - Handling class imbalance with specific techniques
   
3. Evaluation Framework
   - Offline metrics selection (with mathematical definitions)
   - Validation strategy (cross-validation setup)
   - Online evaluation methodology
   - A/B testing design with statistical power analysis

### System Architecture (Always Required)

**Key Elements to Include:**
1. Training Infrastructure
   - Distributed training approach if needed
   - Resource sizing calculations
   - Data pipeline integration points
   - Experiment tracking framework
   
2. Inference Architecture
   - Serving approach (real-time vs batch)
   - Scaling strategy (horizontal vs vertical)
   - Caching mechanisms with hit-rate estimates
   - Failure handling protocols
   
3. Monitoring Strategy
   - Model health metrics (with alert thresholds)
   - Data drift detection mechanisms
   - Feedback loop integration
   - Debugging tooling for production issues

## Common Design Scenarios

### Scenario 1: Product Recommendations

**Example Problem Statement:**
Design a personalized product recommendation system for Amazon's mobile app that improves conversion rate and average order value.

**Key Approach Elements:**

1. Problem Framing
   ```
   Business Objective: Increase conversion rate and average order value
   
   ML Formulation: Ranking problem with personalization
   • Input (X): User features × Item features × Context features
   • Output (Y): Probability of purchase p(purchase|user,item,context)
   • Loss function: Binary cross-entropy with implicit negative sampling
   
   Constraints:
   • Inference latency < 100ms at p99
   • Daily model updates
   • Explainable recommendations for customer service
   ```

2. Data Strategy
   ```
   Data Requirements:
   • User features: Demographics, purchase history, browse history, search queries
   • Item features: Category, price, ratings, attributes, image embeddings
   • Interaction features: View-to-purchase rate, dwell time, recency
   
   Feature Engineering:
   • Temporal features: Decay function f(t) = e^(-λ(t_now - t_interaction))
   • User-item affinity: Cosine similarity between user and item embeddings
   • Price sensitivity: Deviation from average purchase price normalized by user variance
   
   Data Pipeline:
   • Streaming ingest pipeline for real-time events
   • Daily batch processing for aggregate features
   • Feature store with real-time and batch interfaces
   ```

3. Modeling Approach
   ```
   Architecture: Two-stage approach
   
   Retrieval Stage:
   • Two-tower neural network (user tower, item tower)
   • User tower inputs: [u₁, u₂, ..., u_n] → e_u
   • Item tower inputs: [i₁, i₂, ..., i_m] → e_i
   • Similarity score: s(u,i) = e_u·e_i / (||e_u||·||e_i||)
   • Approximate nearest neighbor search (HNSW algorithm)
   
   Ranking Stage:
   • Deep cross-network architecture
   • Explicit feature crossing: (e_u ⊗ e_i)W + b
   • Deep component: MLP(concat(e_u, e_i, context))
   • Multi-task optimization: P(click), P(purchase), E[order_value]
   ```

4. System Architecture
   ```
   Training Pipeline:
   • Daily batch training on Spark
   • Negative sampling ratio 1:10 (positive:negative)
   • Hyperparameter optimization via Bayesian optimization
   
   Inference System:
   • Retrieval: Pre-computed embeddings + ANN service (1000 candidates, <50ms)
   • Ranking: Real-time feature generation + model inference (<50ms)
   • Response composition with diversity injection
   
   Monitoring:
   • Click-through rate (hourly)
   • Feature distribution drift (KL divergence thresholds)
   • Performance segmentation by user cohorts
   • A/B test framework with automatic guardrail metrics
   ```

### Scenario 2: Fraud Detection System

**Example Problem Statement:**
Design a real-time fraud detection system for Amazon Pay transactions that minimizes fraudulent transactions while maintaining high acceptance rates for legitimate users.

**Key Approach Elements:**

1. Problem Framing
   ```
   Business Objective: Minimize fraud losses while maintaining user experience
   
   ML Formulation: Binary classification with cost-sensitive learning
   • Input (X): Transaction features, user history, merchant info, device signals
   • Output (Y): P(fraud|transaction)
   • Loss function: Weighted cross-entropy with FP cost < FN cost
   
   Constraints:
   • Inference latency < 200ms end-to-end
   • Explainable decisions for regulatory compliance
   • High availability (99.99%) requirement
   • Fraud patterns evolve rapidly (concept drift)
   ```

2. Data Strategy
   ```
   Data Requirements:
   • Transaction data: Amount, currency, merchant category, time, location
   • User profile: Account age, transaction history, device history
   • Network features: Graph connections between entities
   • Velocity features: Transaction frequency, amount changes
   
   Feature Engineering:
   • Behavioral biometrics: Typing patterns, mouse movements
   • Temporal patterns: FFT components of transaction time series
   • Graph features: PageRank scores in transaction network
   • Anomaly scores: Deviation from personal spending patterns
   
   Data Pipeline:
   • Real-time event processing (Kafka → Flink)
   • Feature store with sub-100ms access latency
   • Secure storage for sensitive features with access controls
   ```

3. Modeling Approach
   ```
   Architecture: Multi-level detection system
   
   Rule Engine (First Pass):
   • Deterministic rules for known fraud patterns
   • Velocity checks with configurable thresholds
   • Whitelist/blacklist management
   
   ML Models (Second Pass):
   • Gradient Boosted Decision Trees (interpretable features)
   • Neural networks for complex pattern detection
   • Ensemble approach: weighted average with tunable thresholds
   • Automated feature selection via permutation importance
   
   Real-time Learning:
   • Online update for fast adaptation to emerging patterns
   • Bandits for explore-exploit of rule thresholds
   ```

4. System Architecture
   ```
   Training Infrastructure:
   • Daily full retraining + hourly incremental updates
   • Active learning for efficient labeling of edge cases
   • Champion-challenger framework for model comparison
   
   Inference System:
   • Tiered architecture with increasing complexity
   • Cascading models with early exit for clear cases
   • Fallback mechanisms with degraded precision
   • Human-in-loop for borderline cases
   
   Monitoring:
   • Real-time dashboards for fraud analysts
   • Alert system for pattern shifts
   • Performance by segment (transaction size, geography)
   • Investigation tooling for feedback collection
   ```

### Scenario 3: Search Ranking System

**Example Problem Statement:**
Design Amazon's search ranking system to improve relevance and conversion rates for user queries.

**Key Approach Elements:**

1. Problem Framing
   ```
   Business Objective: Improve search relevance and conversion metrics
   
   ML Formulation: Learning to Rank problem
   • Input (X): Query features × Document features × Query-document features
   • Output (Y): Relevance score s(q,d)
   • Loss function: LambdaRank with NDCG optimization
   
   Constraints:
   • Inference latency < 150ms for complete results
   • Support for 100K+ QPS during peak periods
   • Multi-objective optimization (relevance, conversion, revenue)
   ```

2. Data Strategy
   ```
   Data Requirements:
   • Query logs with user interactions
   • Product catalog with structured attributes
   • Historical performance metrics
   • User profiles and personalization signals
   
   Feature Engineering:
   • Query understanding: Entity extraction, intent classification
   • Semantic matching: BERT embeddings for query-product similarity
   • Behavioral signals: CTR, conversion rate, dwell time
   • Business signals: Margin, inventory, shipping speed
   
   Data Pipeline:
   • Near-real-time event processing for interaction data
   • Pre-computation of expensive features
   • Query-independent scoring for partial caching
   ```

3. Modeling Approach
   ```
   Architecture: Multi-stage ranking
   
   Retrieval Stage:
   • Inverted index with BM25 scoring
   • Query expansion with synonyms and related terms
   • Semantic retrieval with dense embeddings
   • Fusion of multiple retrieval strategies
   
   Ranking Stage:
   • LambdaMART base model for interpretability
   • Transformer-based re-ranker for top results
   • Multi-task learning for multiple objectives
   • Personalization layer with user embeddings
   
   Optimization Strategy:
   • Interleaving experiments for fast iteration
   • Counterfactual evaluation with inverse propensity scoring
   • Regularization for fairness and diversity
   ```

4. System Architecture
   ```
   Training Pipeline:
   • Human-labeled judgments for seed data
   • Implicit feedback from user interactions
   • Coordinate ascent for feature weightings
   
   Inference System:
   • Distributed search clusters with replication
   • Caching strategy for popular queries
   • Fallback mechanisms for degraded performance
   • Circuit breakers for dependent services
   
   Monitoring:
   • Query performance tracking by segments
   • A/B testing framework with guardrails
   • Revenue impact analysis
   • Search quality evaluation dashboard
   ```

## System Design Success Checklist

**Before the Interview:**
- [ ] Research Amazon-specific services relevant to ML (SageMaker, Kinesis, etc.)
- [ ] Practice diagramming system architectures quickly and clearly
- [ ] Prepare 2-3 stories about ML systems you've designed/improved
- [ ] Review common ML design patterns and their tradeoffs

**During Problem Framing:**
- [ ] Ask clarifying questions about business objectives
- [ ] Establish clear evaluation metrics (both business and technical)
- [ ] Define constraints explicitly (latency, throughput, resources)
- [ ] Formulate as precise ML problem with mathematical notation

**During Solution Design:**
- [ ] Start with high-level architecture before diving into details
- [ ] Consider multiple approaches before selecting preferred solution
- [ ] Explicitly state tradeoffs for each design decision
- [ ] Incorporate both batch and real-time processing where appropriate
- [ ] Address data quality, validation, and monitoring
- [ ] Design feedback loops for continuous improvement

**Throughout the Interview:**
- [ ] Maintain structured approach with clear sections
- [ ] Use specific numbers (data sizes, latencies, etc.)
- [ ] Draw clear diagrams with components and data flows
- [ ] Connect technical decisions to business objectives
- [ ] Demonstrate awareness of potential failure modes
- [ ] Show ownership mentality by considering operational aspects