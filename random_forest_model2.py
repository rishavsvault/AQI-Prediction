import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE MATH: GINI IMPURITY
# ==========================================
# Formula: 1 - sum(probability^2 of each class)
def calculate_gini(y):
    m = len(y)
    if m == 0: return 0
    # Probability of each class in the split
    probabilities = [np.sum(y == c) / m for c in np.unique(y)]
    return 1 - sum([p**2 for p in probabilities])

# ==========================================
# 2. DATA STRUCTURES
# ==========================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Pollutant index
        self.threshold = threshold  # The value we split at
        self.left = left            # Branch if <= threshold
        self.right = right          # Branch if > threshold
        self.value = value          # Class prediction (for leaf nodes)

# ==========================================
# 3. SPLITTING LOGIC (Best Split Search)
# ==========================================
def get_best_split(X, y, n_features):
    best_gini = 999
    split_idx, split_thresh = None, None
    
    # Feature Subset Selection (The "Random" in Random Forest)
    # We pick a random subset of pollutants to look at for this specific node
    features = np.random.choice(X.shape[1], n_features, replace=False)
    
    for i in features:
        thresholds = np.unique(X[:, i])
        # To speed up training on large CSVs, we can limit threshold checks
        if len(thresholds) > 20:
            thresholds = np.percentile(X[:, i], np.linspace(0, 100, 20))
            
        for t in thresholds:
            left_idx = np.where(X[:, i] <= t)[0]
            right_idx = np.where(X[:, i] > t)[0]
            
            if len(left_idx) == 0 or len(right_idx) == 0: continue
            
            # Calculate Weighted Gini for this split
            g_l, g_r = calculate_gini(y[left_idx]), calculate_gini(y[right_idx])
            w_l, w_r = len(left_idx)/len(y), len(right_idx)/len(y)
            gini = (w_l * g_l) + (w_r * g_r)
            
            if gini < best_gini:
                best_gini, split_idx, split_thresh = gini, i, t
                
    return split_idx, split_thresh

# ==========================================
# 4. BUILDING THE FOREST
# ==========================================
def build_tree(X, y, depth=0, max_depth=10, n_features=None):
    n_samples, n_labels = X.shape[0], len(np.unique(y))
    
    # Stopping Criteria: Max depth reached or only 1 class left
    if depth >= max_depth or n_labels == 1 or n_samples < 5:
        # Majority vote at leaf
        return Node(value=np.bincount(y).argmax())
    
    idx, thr = get_best_split(X, y, n_features)
    if idx is None: return Node(value=np.bincount(y).argmax())
    
    left_indices = np.where(X[:, idx] <= thr)[0]
    right_indices = np.where(X[:, idx] > thr)[0]
    
    left = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth, n_features)
    right = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth, n_features)
    return Node(feature=idx, threshold=thr, left=left, right=right)

class ManualRandomForest:
    def __init__(self, n_trees=5, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = int(np.sqrt(X.shape[1])) # Default: sqrt(number of features)
        for i in range(self.n_trees):
            # BOOTSTRAPPING: Sample data with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = build_tree(X[indices], y[indices], max_depth=self.max_depth, n_features=n_features)
            self.trees.append(tree)
            print(f"  Tree {i+1}/{self.n_trees} trained.")

    def _predict_one(self, x, tree):
        if tree.value is not None: return tree.value
        if x[tree.feature] <= tree.threshold:
            return self._predict_one(x, tree.left)
        return self._predict_one(x, tree.right)

    def predict(self, X):
        # Majority Voting across all trees
        all_tree_preds = np.array([[self._predict_one(x, t) for t in self.trees] for x in X])
        return np.array([np.bincount(row).argmax() for row in all_tree_preds])

# ==========================================
# 5. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading CSV and preparing data...")
df = pd.read_csv("../Data/Air_quality_data.csv")

# Clean Data
df = df.dropna()

# Categorize AQI (0: Good, 1: Moderate, 2: Poor, 3: Severe)
def aqi_to_cat(aqi):
    if aqi <= 50: return 0
    elif aqi <= 150: return 1
    elif aqi <= 300: return 2
    else: return 3
df['AQI_Cat'] = df['AQI'].apply(aqi_to_cat)

# Encode Season & City (Simple Label Encoding for manual math)
df['Month'] = pd.to_datetime(df['Datetime']).dt.month
df['Season_Code'] = df['Month'].apply(lambda x: 0 if x in [12, 1, 2] else (1 if x in [3,4,5] else 2))
city_map = {city: i for i, city in enumerate(df['City'].unique())}
df['City_Code'] = df['City'].map(city_map)

# Select Numeric Features
features_list = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Season_Code', 'City_Code']
X = df[features_list].values
y = df['AQI_Cat'].values

# ==========================================
# 6. EXECUTION
# ==========================================
print("Starting Manual Forest Training...")
# Smaller sample for demonstration to save time
rf = ManualRandomForest(n_trees=5, max_depth=5)
rf.fit(X[:2000], y[:2000]) # Training on first 2000 rows for demo speed

# ==========================================
# 7. VISUALIZATION AND VALIDATION
# ==========================================
# We predict on the first 500 samples to check accuracy
preds = rf.predict(X[:500])
actuals = y[:500]

# Calculate Accuracy manually
accuracy = np.mean(preds == actuals)

plt.figure(figsize=(12, 6))

# Plotting Actual vs Predicted
plt.scatter(range(len(actuals)), actuals, label='Actual AQI Category', alpha=0.6, color='blue', s=50)
plt.scatter(range(len(preds)), preds, label='AI Prediction', marker='x', color='red', s=40)

# Formatting the Chart
plt.title(f"Manual Random Forest: Actual vs Predicted Categories\n(Final Accuracy: {accuracy:.2%})", fontsize=14)
plt.xlabel("Sample Index (First 500 Test Rows)", fontsize=12)
plt.ylabel("AQI Category (0:Good | 1:Moderate | 2:Poor | 3:Severe)", fontsize=12)
plt.yticks([0, 1, 2, 3]) # Ensures only category numbers show on Y-axis
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)

# SAVE the graph so you can share it in chat
plt.savefig('manual_rf_accuracy_plot.png', dpi=300)
print("\n" + "="*30)
print(f"RESULTS SAVED!")
print(f"Accuracy on sample: {accuracy:.2%}")
print("Graph saved as 'manual_rf_accuracy_plot.png'")
print("="*30)

# plt.show() # Commented out to avoid the non-interactive warning