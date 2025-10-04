Problem Statement

In any business, understanding customer behavior is essential for improving marketing strategies, sales, and customer satisfaction. However, customers differ in age, income, and spending habits, making it challenging to treat them all the same way.The goal of this project is to segment customers into distinct groups (clusters) based on their similarities.

This process of dividing customers into smaller, similar groups is known as Customer Segmentation.

Why K-Means Algorithm is Used ??

1) Unsupervised – It doesn’t need labeled data (ideal for customer segmentation).
2) Efficient and Scalable – Works quickly even for large datasets.
3) Simple to Implement – Only needs the number of clusters k and numerical data.
4) Produces Clear Boundaries – Forms well-separated groups that can be visualized easily.

K-Means divides customers into k clusters by minimizing the distance between points and their cluster centroids.
Each cluster represents a group of customers with similar income and spending habits.

CODE EXPLANATION

pandas → For handling data in table (DataFrame) format.

matplotlib.pyplot → For visualizing clusters using scatter plots.

StandardScaler → To standardize (scale) features for better clustering.

KMeans → The algorithm used to form customer clusters.

Real-world data features (like age and income) are on different scales.

K-Means uses Euclidean distance → So scaling is necessary.

StandardScaler() standardizes data (mean = 0, standard deviation = 1).

n_clusters=4: Number of groups you want to form (you can experiment with different values).






This helps businesses understand their audience and personalize marketing strategies effectively.
