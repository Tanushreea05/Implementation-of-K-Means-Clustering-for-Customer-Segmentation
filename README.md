# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the mall customers dataset and check for null values.
2. Use the Elbow Method to determine the optimal number of clusters by running KMeans with clusters ranging from 1 to 10 and calculating the within-cluster sum of squares (WCSS) for each.
3. Plot the WCSS against the number of clusters to identify the "elbow" point.
4. Set the number of clusters to 5 based on the Elbow Method and fit a KMeans model with this cluster count.
5. Predict and assign each customer to a cluster, adding the cluster labels to the dataset.
6. Separate the data into clusters and plot each cluster with different colors based on "Annual Income" and "Spending Score" to visualize customer segments.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Tanushree.A 
RegisterNumber:212223100057  
*/
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="purple",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="gold",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="silver",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="pink",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
## ELBOW METHOD 
![image](https://github.com/user-attachments/assets/da6d8043-4caf-4c13-9bb4-697c801e7699)

## SCATTER PLOT
![image](https://github.com/user-attachments/assets/11daa474-b5e0-4ba1-9bd1-50d70c6d4ca0)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
