# Import necessary libraries
import pandas as pd
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

# Summary statistics
print(customers.describe())
print(products.describe())
print(transactions.describe())

# Distribution of categorical variables
print(customers['Region'].value_counts())
print(products['Category'].value_counts())

# Visualization: Customer distribution by region
plt.figure(figsize=(12, 6))
sns.countplot(x='Region', data=customers)
plt.title('Customer Distribution by Region')
plt.show()

# Visualization: Product categories distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='Category', data=products, order=products['Category'].value_counts().index)
plt.title('Product Categories Distribution')
plt.show()

# Convert TransactionDate to datetime
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Monthly total value of transactions
monthly_transactions = transactions.resample('ME', on='TransactionDate')['TotalValue'].sum()
monthly_transactions.plot(figsize=(12, 6))
plt.title('Monthly Total Value of Transactions')
plt.xlabel('Month')
plt.ylabel('Total Value (USD)')
plt.show()

# Correlation matrix for numerical features
numerical_features = transactions.select_dtypes(include=['number'])
corr_matrix = numerical_features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()