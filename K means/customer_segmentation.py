
# CUSTOMER SEGMENTATION USING K-MEANS


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# STEP 1: Manual Customer Data
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 94, 3, 72, 14, 15, 77, 13, 79, 35, 35, 73, 5, 73, 14],
    'Age': [19, 21, 20, 23, 22, 31, 30, 29, 28, 32, 45, 46, 44, 43, 47, 55, 54, 53, 52, 56]
}

df = pd.DataFrame(data)
print("Dataset preview:")
print(df)


# STEP 2: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# STEP 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# STEP 4: Visualize Clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='Set1', s=100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.show()


# STEP 5: Cluster Analysis
print("\nNumber of customers in each cluster:")
print(df['Cluster'].value_counts())
