import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# 1. LOAD ORIGINAL DATASET (200 Records)
# =========================================================
# =========================================================
# 1. LOAD / GENERATE BASE DATASET (200 Records)
# =========================================================
try:
    url = "https://raw.githubusercontent.com/vipin-k-sharma/Mall_Customers_Segmentation/main/Mall_Customers.csv"
    df_raw = pd.read_csv(url)
except Exception:
    # Local fallback if URL fails
    np.random.seed(42)
    df_raw = pd.DataFrame({
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    })

print(f"Base Dataset Loaded: {len(df_raw)} records")

# =========================================================
# 2. GENERATE EXPANDED UNCLEAN DATA (Multi-Source Simulation)
# =========================================================
# Source A: Customer Demographics (Clean - 200 records)
df_source_a = df_raw[['CustomerID', 'Gender', 'Age']].copy()

# Source B: Customer Behavior Logs (Expanded to 600 records with synthetic noise)
df_source_b = pd.concat(
    [df_raw[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)']] for _ in range(3)], 
    ignore_index=True
)

np.random.seed(42)

# A. Introduce missing values (NaN) across 15% of rows
df_source_b.loc[df_source_b.sample(frac=0.15).index, 'Annual Income (k$)'] = np.nan
df_source_b.loc[df_source_b.sample(frac=0.15).index, 'Spending Score (1-100)'] = np.nan

# B. Inject severe numerical outliers
outlier_idx = df_source_b.sample(n=10).index
df_source_b.loc[outlier_idx[:5], 'Annual Income (k$)'] = [550, 600, 750, 800, 900]
df_source_b.loc[outlier_idx[5:], 'Spending Score (1-100)'] = [-50, -100, 250, 300, 400]

print("\n--- UNCLEAN SOURCE B STATS ---")
print(f"Total Behavior Records: {len(df_source_b)}")
print(f"Duplicate Customer ID Records: {df_source_b.duplicated(subset=['CustomerID']).sum()}")
print("Missing Values Count:\n", df_source_b.isnull().sum())

# =========================================================
# 3. DATA INTEGRATION (MERGING SOURCES)
# =========================================================
df_merged = pd.merge(df_source_a, df_source_b, on='CustomerID', how='right')
print(f"\nMerged Unclean Dataset Total Records: {len(df_merged)} rows")

# =========================================================
# 4. DATA PREPROCESSING PIPELINE
# =========================================================
df_clean = df_merged.copy()

# Step 1: Remove Duplicate Customer Records
df_clean.drop_duplicates(subset=['CustomerID'], keep='first', inplace=True)

# Step 2: Handle Missing Values (Median Imputation)
df_clean['Annual Income (k$)'] = df_clean['Annual Income (k$)'].fillna(df_clean['Annual Income (k$)'].median())
df_clean['Spending Score (1-100)'] = df_clean['Spending Score (1-100)'].fillna(df_clean['Spending Score (1-100)'].median())

# Step 3: Filter Outliers using Interquartile Range (IQR) Method
def remove_outliers(df_in, col_name):
    Q1 = df_in[col_name].quantile(0.25)
    Q3 = df_in[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df_in[(df_in[col_name] >= lower_bound) & (df_in[col_name] <= upper_bound)]

df_clean = remove_outliers(df_clean, 'Annual Income (k$)')
df_clean = remove_outliers(df_clean, 'Spending Score (1-100)')

# Step 4: Feature Standardization (Z-Score Scaling)
X = df_clean[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- CLEANED DATASET SUMMARY ---")
print(f"Final Records Remaining: {len(df_clean)}")
print(f"Missing Values Remaining: {df_clean.isnull().sum().sum()}")

# =========================================================
# 5. K-MEANS CLUSTERING ALGORITHM
# =========================================================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# =========================================================
# 6. DATA VISUALIZATION
# =========================================================
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    data=df_clean, 
    palette='Set1', 
    s=90
)
plt.title('Customer Segments Post-Integration & Preprocessing')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.tight_layout()
plt.show()