# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
customers = pd.read_csv('data/Customers.csv')
products = pd.read_csv('data/Products.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Merge data
customer_product_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Feature engineering: Combine product names for each customer
customer_product_data['ProductName'] = customer_product_data['ProductName'].astype(str)
customer_product_data = customer_product_data.groupby('CustomerID')['ProductName'].apply(' '.join).reset_index()

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(customer_product_data['ProductName'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Recommendation function
def get_lookalike_customers(customer_id, cosine_sim=cosine_sim):
    idx = customer_product_data[customer_product_data['CustomerID'] == customer_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar customers
    lookalike_customers = [(customer_product_data['CustomerID'].iloc[i], score) for i, score in sim_scores]
    return lookalike_customers

# Generate Lookalike.csv
lookalike_dict = {}
customer_ids = customers['CustomerID'][:20]  # First 20 customers

for cust_id in customer_ids:
    lookalike_dict[cust_id] = get_lookalike_customers(cust_id)

lookalike_df = pd.DataFrame(list(lookalike_dict.items()), columns=['cust_id', 'lookalikes'])
lookalike_df.to_csv('FirstName_LastName_Lookalike.csv', index=False)