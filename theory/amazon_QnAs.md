6. Practice Problems for Interview Preparation
Here are detailed walkthroughs for the practice problems, simulating an interview discussion.

6.1. Scenario-Based Problems
Scenario 1: Search Ranking
Interviewer: "Okay, let's talk about search. Can you design a search ranking system for Amazon.com?"

Candidate Response:

"Absolutely. Designing a search ranking system for Amazon is a critical task impacting both customer experience and business revenue. Following Amazon's 'Working Backwards' principle, we start with the customer need: when a customer searches for something, they want to see the most relevant products quickly and easily, leading to a successful purchase or discovery.

1. Problem Understanding & Goals:

Objective: Rank products for a given user query to maximize relevance and likelihood of a positive customer outcome (e.g., click, add-to-cart, purchase).
Scope: We're focusing on the core ranking algorithm after an initial set of candidate products has been retrieved (assuming a separate retrieval step handles finding potentially relevant items from millions of products).
Constraints: High throughput (millions of queries per minute), low latency (results must appear near-instantly), handle diverse queries and product types, potential for personalization.
2. Metrics:

Offline (Technical):
NDCG@k (Normalized Discounted Cumulative Gain): Measures the quality of the ranking based on graded relevance judgments (e.g., exact match, relevant, somewhat relevant). Crucial for capturing the order quality.
MAP (Mean Average Precision): Another list-wise metric, good for binary relevance.
MRR (Mean Reciprocal Rank): Focuses on the rank of the first relevant item. Good for navigational queries.
Online (Business & Customer):
CTR (Click-Through Rate): Percentage of impressions that lead to a click.
CVR (Conversion Rate): Percentage of clicks (or sessions) that lead to a purchase.
Revenue Per Search / Session: Direct business impact.
Add-to-Cart Rate: Intermediate success signal.
Guardrail Metrics: Latency, system availability, diversity of results (avoid showing only one brand/seller).
3. Data & Features:

We need rich features describing the query, the products, and the user (if available).

Query Features:
Query text (n-grams, embeddings).
Query length, query type (navigational, informational, transactional).
Query popularity, historical CTR/CVR for the query.
Product Features:
Product title, description, category, brand (text features, embeddings).
Sales rank, review score/count, price, availability, shipping speed (Prime eligibility).
Image embeddings.
Historical performance (CTR, CVR, sales velocity).
Query-Product Interaction Features (Crucial):
Text Relevance: BM25, TF-IDF scores between query and product text fields. Semantic similarity using embeddings (e.g., cosine similarity between query embedding and product title embedding from BERT or similar models).
Attribute Matching: Does the query contain the product's brand or key attributes?
User Features (for Personalization):
User purchase/browsing history, category affinities.
Demographics (if available and appropriate).
User embeddings.
Contextual Features:
Time of day, device type, location.
4. Model Selection (Learning to Rank - LTR):

Simple heuristics (e.g., sorting by sales rank) are insufficient. LTR models are standard.

Approach: Frame it as learning a scoring function f(query, product, user) such that sorting products by this score yields the optimal ranking.
LTR Methods:
Pointwise: Treat each product independently, predict its relevance score (e.g., using regression) or click probability (classification). Simple but doesn't directly optimize ranking metrics.
Pairwise: Learn a classifier that predicts which product in a pair is more relevant for the query. Examples: RankSVM, RankNet. Better captures relative order.
Listwise: Directly optimize a list-based metric like NDCG. Examples: LambdaMART, ListNet. Often state-of-the-art for ranking.
Chosen Model: LambdaMART (often implemented using XGBoost or LightGBM with a ranking objective like rank:ndcg). It's powerful, handles heterogeneous features well, is relatively efficient, and directly optimizes a relevant ranking metric. Deep learning models (e.g., using Transformers for deep semantic matching) are also powerful but potentially higher latency and complexity. LambdaMART is a strong, practical choice.
5. Training Data:

Need labeled data: (query, list of products, relevance judgments).
Relevance judgments can be explicit (human editors) or implicit (inferred from clicks, purchases - requires careful debiasing as clicks are influenced by position). A common approach is using click data (e.g., clicked item is relevant, skipped items above it are less relevant).
6. Evaluation Strategy:

Offline: Train on one period, validate on the next using temporal splits. Evaluate NDCG, MAP, etc., on the validation set. Perform rigorous hyperparameter tuning (e.g., using SageMaker Hyperparameter Tuning).
Online: A/B testing is essential. Deploy the new ranking model (Candidate B) alongside the current production model (Control A) to a small percentage of traffic. Measure online metrics (CTR, CVR, Revenue, Latency). Gradually ramp up traffic if results are positive and statistically significant.
7. Deployment & System Architecture:

Two-Stage:
Candidate Generation: Fast retrieval system (e.g., inverted index, vector search) finds a few hundred potentially relevant products from millions.
Re-ranking: The LTR model (LambdaMART) scores these candidates using the rich features.
Infrastructure: Use SageMaker for training (distributed if needed) and hosting the model endpoint. Ensure auto-scaling to handle traffic peaks. Consider latency optimizations (feature caching, model quantization, efficient implementation like LightGBM).
Monitoring: Continuously monitor online metrics, latency, error rates. Set up alerts for degradation. Monitor feature drift and model performance decay using tools like SageMaker Model Monitor. Retrain regularly (e.g., daily or weekly) on fresh data.
8. Code Snippet Example (Conceptual Feature Engineering):

python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Assume 'query_embeddings', 'product_embeddings' are precomputed BERT embeddings
# Assume 'df' has columns: 'query', 'product_title', 'sales_rank', 'review_score', 'query_embedding', 'product_embedding'

def create_ranking_features(df):
    # Text Relevance (Simple TF-IDF example)
    vectorizer = TfidfVectorizer()
    all_text = pd.concat([df['query'], df['product_title']])
    vectorizer.fit(all_text)
    query_tfidf = vectorizer.transform(df['query'])
    product_tfidf = vectorizer.transform(df['product_title'])
    # Calculate cosine similarity row-wise (simplified - needs proper alignment)
    # In practice, calculate for each query-product pair
    # df['tfidf_similarity'] = calculate_cosine_similarity(query_tfidf, product_tfidf) # Placeholder

    # Semantic Similarity
    # Ensure embeddings are numpy arrays
    q_emb = np.vstack(df['query_embedding'].values)
    p_emb = np.vstack(df['product_embedding'].values)
    df['semantic_similarity'] = [cosine_similarity(q.reshape(1,-1), p.reshape(1,-1))[0][0] for q, p in zip(q_emb, p_emb)]

    # Existing Product Features (potentially transformed/scaled later)
    df['log_sales_rank'] = np.log1p(df['sales_rank'])
    df['norm_review_score'] = df['review_score'] / 5.0 # Example scaling

    # Combine features... (select relevant columns)
    features = df[['semantic_similarity', 'log_sales_rank', 'norm_review_score']] # Add more features
    return features

# Conceptual XGBoost/LightGBM setup for ranking
# import xgboost as xgb
# model = xgb.XGBRanker(objective='rank:ndcg', ...)
# Need group information (which products belong to which query) for training
# model.fit(X_train, y_train, group=group_train, ...)
9. Challenges & Future Work:

Cold start problem (new products, new queries).
Balancing relevance with diversity and fairness.
Real-time feature updates.
Deeper personalization.
Multi-objective optimization (e.g., relevance vs. profitability).
This provides a comprehensive system design focusing on practical implementation and evaluation."

Scenario 2: Demand Forecasting
Interviewer: "Let's switch gears. How would you build a system to forecast product demand across Amazon's catalog?"

Candidate Response:

"Okay, demand forecasting is fundamental for Amazon's operations, impacting inventory management, supply chain logistics, and pricing. The goal is to predict future demand for potentially millions of SKUs (Stock Keeping Units) accurately and reliably.

1. Problem Understanding & Goals:

Objective: Forecast demand (e.g., units sold) for each SKU at a specific granularity (e.g., daily or weekly) for a defined future horizon (e.g., next 4 weeks).
Customer Impact: Ensure product availability (reduce stockouts) while minimizing excess inventory (reduce holding costs and waste). This directly impacts customer satisfaction and profitability.
Key Requirement: Need not just point forecasts, but also probabilistic forecasts (prediction intervals or quantiles, e.g., P10, P50, P90) to understand uncertainty and make risk-aware inventory decisions.
2. Metrics:

Technical (Point Forecast):
MAPE (Mean Absolute Percentage Error): Intuitive, but problematic for low-volume items or zero actuals.
sMAPE (Symmetric MAPE): Addresses some MAPE issues.
RMSE (Root Mean Squared Error): Sensitive to large errors.
MASE (Mean Absolute Scaled Error): Compares forecast error to a naive baseline (e.g., seasonal naive). Good for comparing across series with different scales.
Technical (Probabilistic Forecast):
Quantile Loss (QL): Measures accuracy at specific quantiles (e.g., wQL - weighted Quantile Loss). Average QL across relevant quantiles (e.g., P10, P50, P90) is a key metric.
Coverage: Does the actual value fall within the predicted interval (e.g., P10-P90) the expected percentage of the time?
Business:
Inventory Holding Cost.
Stockout Rate / Lost Sales.
Forecast Bias (tendency to over/under-forecast).
Inventory Turns.
3. Data & Features:

Target Variable: Historical sales data (units sold per SKU per day/week).
Time Series Features (Derived):
Lag features (sales at t-1, t-7, t-365).
Rolling window statistics (mean, median, std dev over past 7, 30 days).
Time-based features: Day of week, week of year, month, year, holiday flags (including Amazon-specific events like Prime Day), time since product launch.
Related Time Series (Covariates):
Product price history.
Promotional activity flags/discounts.
Inventory levels (can influence sales).
Web traffic/page views for the product.
Sales of related/complementary/substitute products (requires product relationship graph).
Category-level sales trends.
Static Features (Item Metadata):
Product category, brand, color, size, etc. (often used as embeddings or categorical features).
4. Model Selection:

Handling millions of SKUs, complex seasonality, events, and covariates requires scalable models.

Traditional Methods: ARIMA, ETS per SKU. Don't scale well, struggle with covariates and cold starts (new products).
Machine Learning:
Tree-based (XGBoost/LightGBM): Can handle many features, good performance. Requires careful feature engineering to capture time dependencies. Can predict quantiles using quantile regression objective.
Global Models (Deep Learning): Train a single model across all time series. Learns cross-series patterns, handles cold starts better, naturally incorporates covariates.
DeepAR (Proposed): An RNN-based model developed by Amazon, specifically designed for this problem. It learns a distribution for future values, directly providing probabilistic forecasts. Handles covariates, seasonality, cold starts well by learning embeddings for items.
Other DL: LSTNet, N-BEATS, Temporal Fusion Transformers (TFT).
Chosen Model: DeepAR. It's well-suited for Amazon's scale, leverages cross-SKU information, handles covariates, and directly outputs probabilistic forecasts (quantiles), which is crucial for inventory optimization. SageMaker has a built-in DeepAR algorithm.
5. Training Data Preparation (for DeepAR):

Requires specific JSON Lines format:
start: Start timestamp of the series.
target: Array of sales values over time.
feat_dynamic_real: Array of real-valued covariates (e.g., price, rolling averages).
feat_static_cat: Categorical features for the item (e.g., category ID, brand ID).
item_id: Unique identifier for the SKU.
Need to align all time series and covariates. Handle missing values (imputation or model's capability).
6. Evaluation Strategy:

Offline: Use backtesting with temporal splits. Train the model up to time T, forecast for T+1 to T+h (horizon h). Slide the window forward and repeat. Evaluate average Quantile Loss (e.g., across P10, P50, P90) and other metrics (MAPE, RMSE on P50 forecast) over multiple backtest windows.
Online: Deploy the model to generate forecasts. Monitor forecast accuracy against actual sales as they occur. Track downstream business KPIs like stockout rates and inventory levels. Compare against previous forecasting methods or naive baselines. Shadow deployment is often used before fully switching over.
7. Deployment & System Architecture:

Training: Use SageMaker Training Jobs with the built-in DeepAR algorithm. Requires potentially large instances (GPU recommended) for training on millions of series. Training might run weekly or monthly depending on data volume and concept drift.
Inference: Typically a batch process. Use SageMaker Batch Transform daily/weekly to generate forecasts for the required horizon for all SKUs. Store forecasts in a database (e.g., DynamoDB, RDS) or data lake (S3) for consumption by inventory planning systems.
Monitoring: Track forecast accuracy metrics (wQL, MAPE) over time. Monitor input data distributions for drift (using SageMaker Model Monitor or custom checks). Set up alerts for significant accuracy degradation.
8. Code Snippet Example (Conceptual SageMaker DeepAR Setup):

python
import sagemaker
from sagemaker.estimator import Estimator
import boto3

# Assume training/validation data is prepared in JSON Lines format on S3
s3_train_path = "s3://your-bucket/demand-forecast/train/"
s3_validation_path = "s3://your-bucket/demand-forecast/validation/"
s3_output_path = "s3://your-bucket/demand-forecast/output/"

# Get SageMaker execution role and session
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name

# Get DeepAR container image URI
image_uri = sagemaker.image_uris.retrieve("forecasting-deepar", region)

# Configure the DeepAR estimator
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1, # Or more for distributed training
    instance_type='ml.c5.2xlarge', # Or GPU instance like ml.p3.2xlarge
    sagemaker_session=sagemaker_session,
    output_path=s3_output_path
)

# Define DeepAR hyperparameters
# These require careful tuning based on the specific dataset
hyperparameters = {
    "time_freq": "D", # Daily frequency
    "context_length": "30", # Lookback window
    "prediction_length": "28", # Forecast horizon
    "epochs": "100",
    "learning_rate": "0.001",
    "num_layers": "2",
    "num_cells": "50",
    "dropout_rate": "0.1",
    "likelihood": "negative-binomial", # Good for count data
    "embedding_dimension": "20", # For categorical features
    "cardinality": "[AUTO]" # Let SageMaker determine cardinality of cat features
}
estimator.set_hyperparameters(**hyperparameters)

# Define data channels
data_channels = {
    "train": s3_train_path,
    "test": s3_validation_path # Used for validation during training
}

# Start training job
estimator.fit(data_channels)

# After training, create a transformer for batch predictions
# transformer = estimator.transformer(...)
# transformer.transform(...)
9. Challenges & Future Work:

Cold Start: Forecasting for new products with no history (DeepAR helps via embeddings, but still challenging).
High Volatility / Intermittency: Products with very sparse sales.
Cannibalization & Halo Effects: Impact of one product's promotion/stockout on others.
External Events: Unpredictable events impacting demand.
Scalability: Training and inference for millions of SKUs require significant compute resources (AWS infrastructure helps).
This approach provides a robust, scalable system for demand forecasting using state-of-the-art techniques like DeepAR, focusing on the critical need for probabilistic forecasts."

6.2. Coding Exercises
Exercise 1: Feature Engineering
Interviewer: "Let's write some code. Imagine you have a pandas DataFrame representing user actions on an e-commerce site. Can you write a function to engineer some basic features for predicting user behavior, like their next purchase?"

Input Data Description:

Assume a DataFrame df_actions with columns: timestamp (datetime), user_id (int), product_id (int), category (string), action_type (string: 'view', 'add_to_cart', 'purchase'), price (float, only present for 'purchase' actions).

Task: Create features like:

Time-based features (hour, day of week).
User's total purchase count.
User's average purchase value.
User's most frequent category viewed.
Time since the user's last action.
Candidate Solution:

python
import pandas as pd
import numpy as np

def engineer_user_features(df_actions):
    """
    Engineers user-level features from raw action data.

    Args:
        df_actions (pd.DataFrame): DataFrame with columns like
            ['timestamp', 'user_id', 'product_id', 'category',
             'action_type', 'price'].

    Returns:
        pd.DataFrame: DataFrame indexed by user_id with engineered features.
    """
    if df_actions.empty:
        return pd.DataFrame()

    # Ensure timestamp is datetime and sort
    df_actions['timestamp'] = pd.to_datetime(df_actions['timestamp'])
    df_actions = df_actions.sort_values(by=['user_id', 'timestamp'])

    # --- Feature Engineering ---

    # 1. Time-based features (from the *last* action for simplicity here)
    #    In a real scenario, you might aggregate these or use them differently.
    last_action_time = df_actions.groupby('user_id')['timestamp'].last()
    user_features = pd.DataFrame(index=last_action_time.index)
    user_features['last_action_hour'] = last_action_time.dt.hour
    user_features['last_action_dayofweek'] = last_action_time.dt.dayofweek

    # 2. User's total purchase count
    purchase_actions = df_actions[df_actions['action_type'] == 'purchase']
    user_purchase_counts = purchase_actions.groupby('user_id')['action_type'].count()
    user_features['total_purchases'] = user_purchase_counts.reindex(user_features.index, fill_value=0)

    # 3. User's average purchase value
    #    Handle users with no purchases (fill with 0 or mean/median if appropriate)
    user_avg_purchase_value = purchase_actions.groupby('user_id')['price'].mean()
    user_features['avg_purchase_value'] = user_avg_purchase_value.reindex(user_features.index, fill_value=0)
    # Consider filling NaN prices before aggregation if needed

    # 4. User's most frequent category viewed
    view_actions = df_actions[df_actions['action_type'] == 'view']
    # Find the mode (most frequent category) for each user
    # Using lambda that handles empty groups gracefully
    most_freq_category = view_actions.groupby('user_id')['category'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    user_features['most_frequent_view_category'] = most_freq_category.reindex(user_features.index)
    # Handle users with no views (filled with NaN here, could use a placeholder like 'UNKNOWN')

    # 5. Time since the user's last action (relative to a reference time, e.g., now)
    #    Let's assume 'now' is the max timestamp in the dataset for this example
    reference_time = df_actions['timestamp'].max()
    time_since_last_action = (reference_time - last_action_time).dt.total_seconds() / 3600.0 # In hours
    user_features['hours_since_last_action'] = time_since_last_action.reindex(user_features.index)

    # Handle potential NaNs introduced by reindexing or calculations if necessary
    # e.g., user_features.fillna({'most_frequent_view_category': 'UNKNOWN'}, inplace=True)

    return user_features

# --- Example Usage ---
data = {
    'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 11:00:00',
                   '2023-01-02 14:00:00', '2023-01-02 14:10:00', '2023-01-02 15:00:00']),
    'user_id': [1, 1, 1, 2, 2, 1],
    'product_id': [101, 102, 101, 201, 202, 103],
    'category': ['Electronics', 'Books', 'Electronics', 'Home', 'Garden', 'Books'],
    'action_type': ['view', 'view', 'purchase', 'view', 'purchase', 'view'],
    'price': [np.nan, np.nan, 150.0, np.nan, 25.50, np.nan]
}
df_actions_example = pd.DataFrame(data)

user_features_df = engineer_user_features(df_actions_example.copy())
print(user_features_df)
Discussion Points:

This creates user-level aggregate features. For sequence prediction, you might need features at each time step.
Handling missing data (like price for non-purchase actions, or users with no views/purchases) is important.
More advanced features could include rolling window aggregates, category embeddings, time decay effects.
For production, this logic might run in a batch pipeline (e.g., Spark on EMR or AWS Glue) or potentially near real-time using streaming data and a feature store (like SageMaker Feature Store).
Exercise 2: Model Evaluation
Interviewer: "For recommendation systems, standard classification metrics aren't always sufficient. Can you implement functions to calculate Precision@k and Recall@k?"

Task: Write Python functions precision_at_k(y_true, y_pred, k) and recall_at_k(y_true, y_pred, k).

y_true: A list or set of relevant item IDs for a single user.
y_pred: An ordered list of recommended item IDs for that user.
k: The cutoff rank.
Candidate Solution:

python
import numpy as np

def precision_at_k(y_true, y_pred, k):
    """
    Calculates Precision@k for a single user.

    Args:
        y_true (list or set): List/set of relevant item IDs.
        y_pred (list): Ordered list of recommended item IDs.
        k (int): The cutoff rank.

    Returns:
        float: Precision@k score.
    """
    # Ensure y_true is a set for efficient lookup
    y_true_set = set(y_true)

    # Take the top k predictions
    top_k_pred = y_pred[:k]

    # Find the intersection (relevant items among the top k predictions)
    hits = set(top_k_pred) & y_true_set

    # Calculate precision
    if k == 0: # Avoid division by zero if k=0 is passed
        return 0.0
    precision = len(hits) / k
    return precision

def recall_at_k(y_true, y_pred, k):
    """
    Calculates Recall@k for a single user.

    Args:
        y_true (list or set): List/set of relevant item IDs.
        y_pred (list): Ordered list of recommended item IDs.
        k (int): The cutoff rank.

    Returns:
        float: Recall@k score.
    """
    # Ensure y_true is a set for efficient lookup
    y_true_set = set(y_true)
    num_relevant = len(y_true_set)

    # Handle case where there are no relevant items
    if num_relevant == 0:
        return 1.0 if not y_pred[:k] else 0.0 # Or define as 0.0 or NaN based on convention

    # Take the top k predictions
    top_k_pred = y_pred[:k]

    # Find the intersection (relevant items among the top k predictions)
    hits = set(top_k_pred) & y_true_set

    # Calculate recall
    recall = len(hits) / num_relevant
    return recall

# --- Example Usage ---
relevant_items = [101, 105, 201, 305, 400]
recommended_items = [101, 205, 301, 105, 400, 500, 201, 600, 700, 800]
k_value = 5

prec_at_5 = precision_at_k(relevant_items, recommended_items, k=k_value)
rec_at_5 = recall_at_k(relevant_items, recommended_items, k=k_value)

print(f"Relevant Items: {relevant_items}")
print(f"Recommended Items (Top 10): {recommended_items}")
print(f"K = {k_value}")
print(f"Precision@{k_value}: {prec_at_5:.4f}") # Hits: {101, 105, 400}. Precision = 3/5 = 0.6
print(f"Recall@{k_value}: {rec_at_5:.4f}")    # Hits: {101, 105, 400}. Recall = 3/5 = 0.6

k_value = 10
prec_at_10 = precision_at_k(relevant_items, recommended_items, k=k_value)
rec_at_10 = recall_at_k(relevant_items, recommended_items, k=k_value)
print(f"\nK = {k_value}")
print(f"Precision@{k_value}: {prec_at_10:.4f}") # Hits: {101, 105, 400, 201}. Precision = 4/10 = 0.4
print(f"Recall@{k_value}: {rec_at_10:.4f}")    # Hits: {101, 105, 400, 201}. Recall = 4/5 = 0.8
Discussion Points:

These metrics evaluate the top-k items, which is often what users see.
Precision@k focuses on the accuracy within the recommendation list shown.
Recall@k focuses on how many of the truly relevant items were captured in the top-k list.
There's often a trade-off between precision and recall.
To get an overall system performance, you'd average these metrics across many users.
Other important ranking metrics include MAP@k and NDCG@k, which also consider the order of relevant items within the top k.
6.3. Math Problems
Problem 1: Probability Question
Interviewer: "Imagine you ran an A/B test for a new feature. The control group (A) had 1000 users and 100 conversions. The treatment group (B) had 1100 users and 120 conversions. How would you determine the probability that the treatment group's true conversion rate is higher than the control group's?"

Candidate Response:

"This is a classic problem comparing two proportions. While we could use a frequentist hypothesis test (like a z-test for two proportions) to get a p-value, the question asks for the probability that the treatment rate (p_B) is greater than the control rate (p_A). This naturally leads to a Bayesian approach.

1. Modeling:

We can model the true underlying conversion rates, p_A and p_B, as random variables.
The number of conversions in each group follows a Binomial distribution:
Conversions_A ~ Binomial(n_A, p_A), where n_A = 1000, k_A = 100
Conversions_B ~ Binomial(n_B, p_B), where n_B = 1100, k_B = 120
We need priors for p_A and p_B. A common non-informative prior for a probability is the Beta(1, 1) distribution (which is equivalent to a Uniform(0, 1) distribution). Let's use this.
The Beta distribution is the conjugate prior for the Binomial likelihood, meaning the posterior distribution will also be a Beta distribution.
2. Calculating Posteriors:

The posterior distribution for a Beta prior Beta(α, β) and Binomial likelihood Binomial(n, k) is Beta(α + k, β + n - k).
Posterior for p_A: Beta(1 + k_A, 1 + n_A - k_A) = Beta(1 + 100, 1 + 1000 - 100) = Beta(101, 901)
Posterior for p_B: Beta(1 + k_B, 1 + n_B - k_B) = Beta(1 + 120, 1 + 1100 - 120) = Beta(121, 981)
3. Finding P(p_B > p_A | data):

We now have the probability distributions representing our updated beliefs about p_A and p_B after observing the data. We want to calculate the probability that a random draw from the posterior of p_B is greater than a random draw from the posterior of p_A.

This integral P(p_B > p_A) = ∫∫ I(p_B > p_A) * P(p_A|data) * P(p_B|data) dp_A dp_B doesn't have a simple closed-form solution. However, we can easily estimate it using Monte Carlo simulation:

Draw a large number of samples (e.g., N = 100,000) from the posterior distribution of p_A (Beta(101, 901)).
Draw the same number of samples from the posterior distribution of p_B (Beta(121, 981)).
Compare the samples pairwise: count how many times sample_B > sample_A.
The probability P(p_B > p_A | data) is estimated as (Count of times sample_B > sample_A) / N.
4. Implementation (Python):

python
import numpy as np
from scipy.stats import beta

# Parameters
n_A, k_A = 1000, 100
n_B, k_B = 1100, 120

# Posterior parameters (alpha = 1+k, beta = 1+n-k)
alpha_A, beta_A = 1 + k_A, 1 + n_A - k_A
alpha_B, beta_B = 1 + k_B, 1 + n_B - k_B

# Number of samples for Monte Carlo simulation
N_samples = 100000

# Draw samples from posterior distributions
samples_A = beta.rvs(alpha_A, beta_A, size=N_samples)
samples_B = beta.rvs(alpha_B, beta_B, size=N_samples)

# Calculate the proportion of times sample_B > sample_A
prob_B_gt_A = np.mean(samples_B > samples_A)

# Also useful: calculate expected lift and distribution of lift
lift_samples = (samples_B - samples_A) / samples_A
expected_lift = np.mean(lift_samples)
prob_lift_positive = np.mean(lift_samples > 0) # Should be same as prob_B_gt_A

print(f"Posterior A: Beta({alpha_A}, {beta_A})")
print(f"Posterior B: Beta({alpha_B}, {beta_B})")
print(f"\nEstimated P(p_B > p_A | data): {prob_B_gt_A:.4f}")
print(f"Estimated Expected Relative Lift: {expected_lift:.4f}")

# Frequentist check (z-test for proportions)
p_A_obs = k_A / n_A
p_B_obs = k_B / n_B
p_pool = (k_A + k_B) / (n_A + n_B)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z_score = (p_B_obs - p_A_obs) / se_pool
p_value = 1 - stats.norm.cdf(z_score) # One-sided test
print(f"\n--- Frequentist Check ---")
print(f"Observed p_A: {p_A_obs:.4f}, Observed p_B: {p_B_obs:.4f}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value (one-sided): {p_value:.4f}")
5. Interpretation:

The simulation result (likely around 80-90%, let's say 0.85 for discussion) gives us the probability that the treatment group's underlying conversion rate is truly higher than the control's, given the observed data and our prior assumptions. This is often more intuitive for business stakeholders than a p-value. We can also calculate the expected lift and the probability of the lift being greater than some meaningful threshold.

The frequentist p-value gives the probability of observing data at least as extreme as ours, assuming the null hypothesis (p_A = p_B) is true. A small p-value (e.g., < 0.05) would lead us to reject the null hypothesis. The Bayesian approach directly answers the probability question asked."

Problem 2: Optimization Question
Interviewer: "Let's consider a simple inventory problem. You need to decide how many units of a product to order for the upcoming week. You know the cost per unit (c), the selling price (p), and a salvage value (s) for unsold units. You also have a forecast for demand, let's say it follows a Normal distribution with mean mu and standard deviation sigma. How do you determine the optimal order quantity q to maximize expected profit?"

Candidate Response:

"This is a classic inventory management problem known as the Newsvendor Problem (or single-period inventory model). The core challenge is balancing the cost of ordering too much (overage cost) against the cost of ordering too little (underage cost).

1. Define Costs:

Cost of Overage (C_o): The cost incurred for each unit ordered that is not sold. This is the ordering cost minus the salvage value, plus any holding cost (let's assume holding cost h per unit for simplicity, though it wasn't explicitly mentioned initially).
C_o = c - s + h (If h=0, then C_o = c - s)
Cost of Underage (C_u): The opportunity cost incurred for each unit of demand that cannot be met because not enough was ordered. This is the lost profit from a potential sale.
C_u = p - c
2. Objective:

Maximize the expected profit. We can achieve this by finding the order quantity q where the expected marginal gain of ordering one more unit equals the expected marginal loss.

3. Marginal Analysis:

Consider ordering the q-th unit.

This unit will be sold if Demand D >= q. The profit gained is p - c = C_u. The probability of this is P(D >= q).
This unit will not be sold if Demand D < q. The loss incurred is c - s + h = C_o. The probability of this is P(D < q).
We should order the q-th unit as long as the expected gain is greater than or equal to the expected loss: P(D >= q) * C_u >= P(D < q) * C_o

4. Finding the Optimal Quantity (Critical Fractile):

Let F(q) = P(D <= q) be the cumulative distribution function (CDF) of demand. Then P(D < q) is approximately F(q) (or exactly F(q-1) for discrete demand, but we'll use the continuous approximation with the Normal distribution). And P(D >= q) = 1 - P(D < q) ≈ 1 - F(q).

Substituting into the inequality: (1 - F(q)) * C_u >= F(q) * C_o C_u - F(q) * C_u >= F(q) * C_o C_u >= F(q) * (C_u + C_o) F(q) <= C_u / (C_u + C_o)

We want to find the largest q that satisfies this. The optimal point q* is where the expected profit from the next unit is zero, which occurs when: F(q*) = C_u / (C_u + C_o)

This ratio C_u / (C_u + C_o) is called the critical fractile or critical ratio. It represents the service level we are aiming for – the probability that demand will be less than or equal to our order quantity.

5. Solution for Normal Demand:

Given that Demand D ~ Normal(mu, sigma), we need to find q* such that: P(D <= q*) = Critical Fractile

We standardize the variable: Z = (q* - mu) / sigma. P(Z <= (q* - mu) / sigma) = Critical Fractile Let Phi(z) be the standard normal CDF. We need to find the z-score z* such that Phi(z*) = Critical Fractile. We can find this using a standard normal table or scipy.stats.norm.ppf().

Once we have z*: (q* - mu) / sigma = z* q* = mu + z* * sigma

Since we must order an integer number of units, we typically round q* up or down based on context or exact calculation if the profit function is analyzed more closely near this value. Often, rounding to the nearest integer or rounding up is used.

6. Example Calculation:

Suppose: p = $50, c = $20, s = $5, h = $1
Demand D ~ Normal(mu=100, sigma=30)
Calculate costs:
C_u = p - c = 50 - 20 = $30
C_o = c - s + h = 20 - 5 + 1 = $16
Calculate Critical Fractile:
CF = C_u / (C_u + C_o) = 30 / (30 + 16) = 30 / 46 ≈ 0.6522
Find the z-score z* such that Phi(z*) = 0.6522.
Using scipy.stats.norm.ppf(0.6522), we get z* ≈ 0.39.
Calculate optimal quantity q*:
q* = mu + z* * sigma = 100 + 0.39 * 30 = 100 + 11.7 = 111.7
Optimal Order Quantity: Rounding up, we should order 112 units.
7. Gradient-Based Approach (Mention if asked):

While the critical fractile method is standard, one could formulate the expected profit function E[Profit(q)] explicitly: E[Profit(q)] = E[p * min(q, D) - c * q - C_o * max(0, q - D)] This involves integrals over the probability density function of D. We could then find the derivative of E[Profit(q)] with respect to q (using Leibniz integral rule) and set it to zero to find the maximum. This would lead back to the same critical fractile condition but is mathematically more involved than the marginal analysis.

8. Discussion:

This model assumes demand distribution is known, costs are fixed, and it's a single period.
Extensions handle multi-period inventory, lead times, unknown demand distributions (requiring forecasting first), capacity constraints, etc.
The critical fractile provides a clear trade-off: higher C_u (high profit margin) or lower C_o (low holding/salvage loss) leads to a higher critical fractile and thus ordering more units (higher service level).
This structured approach using marginal analysis and the critical fractile provides the optimal solution for the classic Newsvendor problem."

