import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.csv")

x=data.iloc[:,[2,58]].values

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,14):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#plt.plot(range(1,14),wcss)
#plt.show()
#5-7 ideal

kmeans=KMeans(n_clusters=5)
yhead=kmeans.fit_predict(x)
plt.scatter(x[yhead==0,0],x[yhead==0,1],color="red")
plt.scatter(x[yhead==1,0],x[yhead==1,1],color="blue")
plt.scatter(x[yhead==2,0],x[yhead==2,1],color="green")
plt.scatter(x[yhead==3,0],x[yhead==3,1],color="yellow")
plt.scatter(x[yhead==4,0],x[yhead==4,1],color="orange")
plt.show()