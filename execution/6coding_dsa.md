# Coding & Data Structures/Algorithms Round

## Format
- 60-minute interview focused on coding and problem-solving
- Expectation: 2-3 problems of medium-hard difficulty
- Focus on correctness, efficiency, and clean code
- Mostly standard DSA with occasional ML algorithm implementation

## Common Problem Categories

### Array and String Manipulation (High Likelihood)

**Problem: Longest Substring Without Repeating Characters**

```
Given a string s, find the length of the longest substring without repeating characters.

Example:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

**Solution Strategy:**
```python
def lengthOfLongestSubstring(s: str) -> int:
    # Sliding window approach
    char_index = {}  # Track index of each character
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If we've seen this character after our current window started
        if char in char_index and char_index[char] >= start:
            # Move the start pointer to position after the repeated character
            start = char_index[char] + 1
        else:
            # Update max_length if current window is larger
            max_length = max(max_length, end - start + 1)
        
        # Update the index of current character
        char_index[char] = end
    
    return max_length
```

**Time Complexity:** O(n) where n is the length of the string
**Space Complexity:** O(min(m, n)) where m is the size of the character set

**Key Insights:**
- Sliding window is efficient for substring problems
- Hash map provides O(1) lookups for previous occurrences
- Careful handling of window boundaries is critical

---

**Problem: Merge Intervals**

```
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
```

**Solution Strategy:**
```python
def merge(intervals):
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        # If merged is empty or current interval doesn't overlap with previous
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Update end time of previous interval if current interval overlaps
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged
```

**Time Complexity:** O(n log n) due to sorting
**Space Complexity:** O(n) for the output array

**Key Insights:**
- Sorting first simplifies the merging process
- Greedy approach works because we process intervals in order
- Consider edge cases: empty input, single interval, no overlaps

### Tree and Graph Algorithms (High Likelihood)

**Problem: Lowest Common Ancestor of a Binary Tree**

```
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself)."

Example:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
```

**Solution Strategy:**
```python
def lowestCommonAncestor(root, p, q):
    # Base cases
    if not root or root == p or root == q:
        return root
    
    # Search in left and right subtrees
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    # If both left and right have values, root is the LCA
    if left and right:
        return root
    
    # Otherwise, return the non-null value
    return left if left else right
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree (for recursion stack)

**Key Insights:**
- Postorder traversal allows bottom-up identification of LCA
- Result either bubbles up from subtrees or is the current node
- Works for both binary search trees and generic binary trees

---

**Problem: Course Schedule (Topological Sort)**

```
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
Return true if you can finish all courses. Otherwise, return false.

Example:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.
```

**Solution Strategy:**
```python
def canFinish(numCourses, prerequisites):
    # Build adjacency list
    graph = [[] for _ in range(numCourses)]
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # 0 = unvisited, 1 = visiting (in current DFS path), 2 = visited
    status = [0] * numCourses
    
    def hasCycle(course):
        # If we're visiting this node in the current path, we found a cycle
        if status[course] == 1:
            return True
        # If we've already processed this node, no need to do it again
        if status[course] == 2:
            return False
        
        # Mark as currently visiting
        status[course] = 1
        
        # Check all prerequisites for cycles
        for prereq in graph[course]:
            if hasCycle(prereq):
                return True
        
        # Mark as fully visited
        status[course] = 2
        return False
    
    # Check each course for cycles
    for course in range(numCourses):
        if hasCycle(course):
            return False
    
    return True
```

**Time Complexity:** O(V + E) where V is the number of courses and E is the number of prerequisites
**Space Complexity:** O(V + E) for the adjacency list and status array

**Key Insights:**
- This is a cycle detection problem in a directed graph
- We can solve it using DFS with visit tracking
- If no cycles, a valid topological order exists

### Dynamic Programming (Medium Likelihood)

**Problem: Coin Change**

```
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.

Example:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
```

**Solution Strategy:**
```python
def coinChange(coins, amount):
    # Initialize dp array with amount+1 (which is greater than any possible result)
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed to make amount 0
    
    # For each amount from 1 to target
    for current_amount in range(1, amount + 1):
        # Try each coin denomination
        for coin in coins:
            # If this coin can contribute to current amount
            if coin <= current_amount:
                # Update with minimum number of coins
                dp[current_amount] = min(dp[current_amount], dp[current_amount - coin] + 1)
    
    # If dp[amount] was not updated, no solution exists
    return dp[amount] if dp[amount] != amount + 1 else -1
```

**Time Complexity:** O(amount * n) where n is the number of coin denominations
**Space Complexity:** O(amount) for the dp array

**Key Insights:**
- Bottom-up DP builds solution incrementally
- Recurrence relation: dp[i] = min(dp[i], dp[i-coin] + 1)
- Initialization with amount+1 serves as "infinity" marker

---

**Problem: Maximum Subarray Sum**

```
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

**Solution Strategy:**
```python
def maxSubArray(nums):
    # Initialize variables
    max_so_far = nums[0]
    current_max = nums[0]
    
    # Iterate starting from second element
    for i in range(1, len(nums)):
        # Either extend previous subarray or start new one
        current_max = max(nums[i], current_max + nums[i])
        # Update global maximum
        max_so_far = max(max_so_far, current_max)
    
    return max_so_far
```

**Time Complexity:** O(n) 
**Space Complexity:** O(1)

**Key Insights:**
- This is Kadane's algorithm
- Key insight: at each position, decide whether to extend previous subarray or start new one
- Dynamic programming with constant space optimization

### Machine Learning Algorithm Implementation (Medium Likelihood)

**Problem: Implement K-means Clustering**

```
Implement the k-means clustering algorithm from scratch. Your function should take a dataset X, number of clusters k, and maximum iterations, and return cluster assignments and centroids.

Example:
Input: X = [[1,2], [1,4], [1,0], [4,2], [4,4], [4,0]], k = 2
Output: Clusters and centroids representing the two groups: [[1,2], [1,4], [1,0]] and [[4,2], [4,4], [4,0]]
```

**Solution Strategy:**
```python
import numpy as np

def kmeans(X, k, max_iters=100):
    # Convert input to numpy array
    X = np.array(X)
    n_samples, n_features = X.shape
    
    # Randomly initialize k centroids from the data points
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]
    
    # Initialize cluster assignments
    clusters = np.zeros(n_samples)
    
    for _ in range(max_iters):
        # Assign each point to nearest centroid
        for i in range(n_samples):
            # Calculate distances to each centroid
            distances = np.sqrt(np.sum((centroids - X[i])**2, axis=1))
            # Assign to closest centroid
            clusters[i] = np.argmin(distances)
        
        # Store old centroids for convergence check
        old_centroids = centroids.copy()
        
        # Update centroids based on new assignments
        for j in range(k):
            # Get points in this cluster
            cluster_points = X[clusters == j]
            # If cluster has points, update centroid
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
        
        # Check for convergence (centroids no longer move)
        if np.all(old_centroids == centroids):
            break
    
    return clusters, centroids
```

**Time Complexity:** O(n_samples * k * max_iters * n_features)
**Space Complexity:** O(n_samples + k * n_features)

**Key Insights:**
- Random initialization can affect final results
- Convergence is guaranteed but may reach local optimum
- Need to handle possible empty clusters

---

**Problem: Implement Logistic Regression with Gradient Descent**

```
Implement logistic regression with gradient descent optimization from scratch. Your function should take training data X, labels y, learning rate, and number of iterations, and return trained weights.

Example:
Input: X = [[1,2], [2,3], [3,4], [5,6], [6,7]], y = [0, 0, 0, 1, 1], learning_rate = 0.1, iterations = 1000
Output: Weights that can classify the points
```

**Solution Strategy:**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    # Add intercept term
    X = np.c_[np.ones(X.shape[0]), X]
    # Convert arrays to numpy format
    X = np.array(X)
    y = np.array(y)
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    
    # Gradient descent
    for _ in range(iterations):
        # Calculate predictions
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        
        # Calculate error
        error = predictions - y
        
        # Calculate gradient
        gradient = np.dot(X.T, error) / len(y)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Optional: Calculate and print cost function
        # cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions + 1e-9))
        # print(f"Cost: {cost}")
    
    return weights

def predict(X, weights):
    # Add intercept term if not present
    if X.shape[1] == len(weights) - 1:
        X = np.c_[np.ones(X.shape[0]), X]
    
    # Calculate probability
    z = np.dot(X, weights)
    probability = sigmoid(z)
    
    # Convert to binary prediction
    return (probability >= 0.5).astype(int)
```

**Time Complexity:** O(iterations * n_samples * n_features)
**Space Complexity:** O(n_features)

**Key Insights:**
- Gradient descent optimizes log likelihood function
- Need to handle numerical stability for log(0)
- Learning rate selection affects convergence speed and stability

### Binary Search and Sorting (Medium Likelihood)

**Problem: Search in Rotated Sorted Array**

```
There is an integer array nums sorted in ascending order (with distinct values).
Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed).
Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
You must write an algorithm with O(log n) runtime complexity.

Example:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Solution Strategy:**
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # If we found the target
        if nums[mid] == target:
            return mid
        
        # Check if the left half is sorted
        if nums[left] <= nums[mid]:
            # Target is in the sorted left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half must be sorted
        else:
            # Target is in the sorted right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

**Key Insights:**
- Modified binary search that handles rotation
- One half must always be sorted
- Determine which half is sorted and whether target is in that half

---

**Problem: Find First and Last Position of Element in Sorted Array**

```
Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
If target is not found in the array, return [-1, -1].
You must write an algorithm with O(log n) runtime complexity.

Example:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Solution Strategy:**
```python
def searchRange(nums, target):
    # Helper function to find leftmost or rightmost occurrence
    def binarySearch(leftmost):
        left, right = 0, len(nums) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:  # nums[mid] == target
                result = mid  # Potential result
                
                if leftmost:
                    # Continue searching in left half for leftmost occurrence
                    right = mid - 1
                else:
                    # Continue searching in right half for rightmost occurrence
                    left = mid + 1
        
        return result
    
    # Find leftmost and rightmost occurrences
    left_idx = binarySearch(leftmost=True)
    right_idx = binarySearch(leftmost=False)
    
    return [left_idx, right_idx]
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

**Key Insights:**
- Two binary searches: one for leftmost, one for rightmost
- Reusable binary search function with a flag parameter
- When we find the target, we don't immediately return but continue searching

## ML-Specific Coding Problems (Low Likelihood)

**Problem: Implement a Decision Tree Node Split**

```
Implement a function to determine the best attribute and split point for a decision tree based on information gain. The function should take a dataset, current attribute values, and target values, and return the best attribute, split value, and information gain.

Example:
Input: X = [[2,3], [3,4], [2,4], [3,3]], y = [0, 0, 1, 1]
Output: (0, 2.5, 0.311) meaning attribute index 0, split point 2.5, information gain 0.311
```

**Solution Strategy:**
```python
import numpy as np
from collections import Counter

def entropy(y):
    """Calculate entropy of a target array."""
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

def information_gain(y, y_left, y_right):
    """Calculate information gain from a split."""
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    return entropy(y) - (p_left * entropy(y_left) + p_right * entropy(y_right))

def find_best_split(X, y):
    """Find best attribute and split point using information gain."""
    best_gain = -1
    best_attr = None
    best_value = None
    
    n_samples, n_features = X.shape
    
    for feature_idx in range(n_features):
        # Get unique values in sorted order
        values = sorted(set(X[:, feature_idx]))
        
        # Try different split points
        for i in range(len(values) - 1):
            split_value = (values[i] + values[i + 1]) / 2
            
            # Split the data
            left_indices = X[:, feature_idx] <= split_value
            right_indices = ~left_indices
            
            y_left = y[left_indices]
            y_right = y[right_indices]
            
            # Skip if split doesn't divide the data
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            # Calculate information gain
            gain = information_gain(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_attr = feature_idx
                best_value = split_value
    
    return best_attr, best_value, best_gain
```

**Time Complexity:** O(n_samples * n_features * log(n_samples))
**Space Complexity:** O(n_samples)

**Key Insights:**
- Information gain quantifies split quality
- Consider all possible attributes and split points
- Entropy calculation involves log operations
- Edge case handling for empty splits is important

---

**Problem: Implement a Mini-batch Gradient Descent**

```
Implement mini-batch gradient descent for linear regression. Your function should take training data X, targets y, batch size, learning rate, and number of epochs, and return trained weights.

Example:
Input: X = [[1,2], [2,3], [3,4], [4,5], [5,6]], y = [3, 5, 7, 9, 11], batch_size = 2, learning_rate = 0.01, epochs = 100
Output: Weights close to [1, 1] (since y = x1 + x2)
```

**Solution Strategy:**
```python
import numpy as np

def minibatch_gradient_descent(X, y, batch_size=2, learning_rate=0.01, epochs=100):
    # Add intercept term
    X = np.c_[np.ones(X.shape[0]), X]
    X, y = np.array(X), np.array(y)
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    # Create indices for batches
    indices = np.arange(n_samples)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data at each epoch
        np.random.shuffle(indices)
        
        # Process mini-batches
        for start_idx in range(0, n_samples, batch_size):
            # Get batch indices
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Extract batch data
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Calculate predictions
            predictions = np.dot(X_batch, weights)
            
            # Calculate error
            errors = predictions - y_batch
            
            # Calculate gradient
            gradient = np.dot(X_batch.T, errors) / len(batch_indices)
            
            # Update weights
            weights -= learning_rate * gradient
        
        # Optional: calculate and print loss after each epoch
        # predictions = np.dot(X, weights)
        # loss = np.mean((predictions - y) ** 2) / 2
        # print(f"Epoch {epoch+1}, Loss: {loss}")
    
    return weights
```

**Time Complexity:** O(epochs * n_samples * n_features)
**Space Complexity:** O(n_features + batch_size)

**Key Insights:**
- Mini-batch combines stability of batch GD with efficiency of SGD
- Shuffling data prevents cyclical patterns
- Batch size is a hyperparameter affecting convergence
- Learning rate must be appropriate for stability

## Coding Round Success Checklist

- [ ] Clarify problem requirements before coding
- [ ] Discuss approach and complexity before implementation
- [ ] Write clean, readable code with proper variable names
- [ ] Consider edge cases (empty input, single element, etc.)
- [ ] Test solution with example inputs
- [ ] Analyze time and space complexity
- [ ] Optimize solution if initial approach is inefficient
- [ ] Communicate thought process throughout
- [ ] Be prepared to modify approach based on interviewer feedback
- [ ] Handle errors and exceptions appropriately
- [ ] For ML problems, discuss tradeoffs in algorithm selection