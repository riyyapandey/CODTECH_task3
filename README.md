Name: Riya Pandey

Company: CODTECH IT SOLUTIONS

ID: CT6WDS1269

Domain: Data Analytics

Duration: July to August 2024

Mentor: Muzammil AHMED


Overview of this Project

Project: Customer Segmentation and Analysis

This project involves clustering customers using the Mall_Customers dataset to identify distinct segments based on their age, annual income, and spending score. Here's a detailed overview of the project:

Objectives:
1. Load and Preprocess the Data:
   - Load the dataset into a pandas DataFrame.
   - Standardize the features for clustering.

2. Determine Optimal Number of Clusters:
   - Use the Elbow Method to plot inertia (sum of squared distances to the nearest cluster center) against the number of clusters.
   - Use the Silhouette Score to evaluate the quality of the clustering.

3. Perform K-means Clustering:
   - Select the optimal number of clusters based on the Elbow Curve and Silhouette Score.
   - Apply K-means clustering to segment the customers.

4. Analyze Segments:
   - Assign cluster labels to the original data.
   - Analyze the characteristics of each segment (e.g., average age, annual income, spending score, and count of customers per cluster).

Steps:

1. Load the Dataset:
   - Simulate the Mall_Customers dataset with random values for CustomerID, Gender, Age, Annual Income, and Spending Score.

   
   data = {
       'CustomerID': np.arange(1, 201),
       'Gender': np.random.choice(['Male', 'Female'], size=200),
       'Age': np.random.randint(18, 70, size=200),
       'Annual Income (k$)': np.random.randint(15, 150, size=200),
       'Spending Score (1-100)': np.random.randint(1, 100, size=200)
   }
   df = pd.DataFrame(data)
   ```

2. Data Preprocessing:
   - Extract relevant features (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
   - Standardize the features using `StandardScaler`.

   
   X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. Determine Optimal Number of Clusters:
   - Implement the function `find_optimal_clusters` to calculate inertia and silhouette scores for different cluster counts.
   - Plot the Elbow Curve and Silhouette Scores.

   
   def find_optimal_clusters(data, max_k):
       inertias = []
       sil_scores = []
       for k in range(2, max_k+1):
           kmeans = KMeans(n_clusters=k, random_state=42)
           kmeans.fit(data)
           inertias.append(kmeans.inertia_)
           if k > 1:
               sil_scores.append(silhouette_score(data, kmeans.labels_))
       return inertias, sil_scores

   max_clusters = 10
   inertias, sil_scores = find_optimal_clusters(X_scaled, max_clusters)
   ```

4. Plotting:
   - Create plots for the Elbow Curve and Silhouette Scores.

   
   plt.figure(figsize=(12, 5))
   plt.subplot(1, 2, 1)
   plt.plot(range(2, max_clusters+1), inertias, marker='o', linestyle='--')
   plt.xlabel('Number of Clusters')
   plt.ylabel('Inertia')
   plt.title('Elbow Curve')

   plt.subplot(1, 2, 2)
   plt.plot(range(2, max_clusters+1), sil_scores, marker='o', linestyle='--')
   plt.xlabel('Number of Clusters')
   plt.ylabel('Silhouette Score')
   plt.title('Silhouette Score')

   plt.tight_layout()
   plt.show()
   ```

5. Select Optimal Number of Clusters and Perform Clustering:
   - Based on the plots, choose the optimal number of clusters (e.g., 5).
   - Apply K-means clustering.

   
   optimal_clusters = 5
   kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
   kmeans.fit(X_scaled)
   df['Cluster'] = kmeans.labels_
   ```

6. Analyze Segments:
   - Group the data by cluster and calculate the mean of each feature and the count of customers in each cluster.

   
   segment_analysis = df.groupby('Cluster').agg({
       'Age': 'mean',
       'Annual Income (k$)': 'mean',
       'Spending Score (1-100)': 'mean',
       'CustomerID': 'count'
   }).reset_index()
   print(segment_analysis)
   ```

Results:
The results will provide insights into different customer segments, which can be used for targeted marketing and personalized customer experiences. The segment analysis will include average age, annual income, spending score, and the number of customers in each cluster.

