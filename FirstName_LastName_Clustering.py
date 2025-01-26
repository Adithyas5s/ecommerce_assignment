# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv('data/Customers.csv')
products = pd.read_csv('data/Products.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Display basic information
print(customers.info())
print(products.info())
print(transactions.info())

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Drop duplicates
customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)

# Merge data
customer_product_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID', suffixes=('_trans', '_prod'))

# Inspect the merged data
print(customer_product_data.info())

# Aggregate transaction data
customer_agg = customer_product_data.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum',
    'Price_prod': 'mean'  # Use the correct Price column from Products.csv
}).reset_index()

# Combine with customer data
customer_data = customer_agg.merge(customers, on='CustomerID')

# Select features for clustering
features = ['Quantity', 'TotalValue', 'Price_prod']
customer_features = customer_data[features]

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(customer_features)

# Add cluster labels to customer data
customer_data['Cluster'] = clusters

# Clustering metrics
db_index = davies_bouldin_score(customer_features, clusters)
silhouette_avg = silhouette_score(customer_features, clusters)

print(f'DB Index: {db_index}')
print(f'Silhouette Score: {silhouette_avg}')

# Visualize clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Quantity', y='TotalValue', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Total Quantity Purchased')
plt.ylabel('Total Value of Transactions (USD)')
plt.show()