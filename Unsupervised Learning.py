import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

%matplotlib inline

data = pd.read_csv('C:\\Users\\user\\Downloads\\assignments\\Module 9\\customers.csv')

data.head()

data

data.isnull().sum()

data.info()

data.describe()

data.plot.scatter(x='Age', y='Spending Score (1-100)')


data0 = data[['Age' , 'Spending Score (1-100)']].values

i = []
for n in range(1 , 11):
    algorithm = KMeans(n_clusters = n)
    algorithm.fit(data0)
    i.append(algorithm.inertia_)


i

pd.DataFrame(i).plot()


KMeans_Clustering = KMeans(n_clusters = 5)
KMeans_Clustering.fit(data0)
labels = KMeans_Clustering.labels_


data0= data[['Age', 'Spending Score (1-100)']]

data0

labeled_data = data0.assign(cluster=labels)

labeled_data

labeled_data.plot.scatter(x='Age', y='Spending Score (1-100)', c='cluster', colormap='jet')

