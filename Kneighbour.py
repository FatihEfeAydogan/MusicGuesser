import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Veri okuma
data = pd.read_csv("C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.csv")

# İlk sütunu veri setinden çıkar
data = data.drop(columns=['filename'])

# Etiketlerin one-hot encoding ile dönüştürülmesi
label_encoder = LabelEncoder()
y1_encoded = label_encoder.fit_transform(data['label'])

# Özellik ve hedef veri setleri
x1 = data.iloc[:, :-1]  # Son sütunu almadık
y1 = y1_encoded

# Veri setinin eğitim ve test kümelerine bölünmesi
xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.3, random_state=43)

# Verilerin standartlaştırılması
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)

# KNN sınıflandırıcının oluşturulması ve eğitilmesi
knn = KNeighborsClassifier(5)
knn.fit(xtrain1, ytrain)

# Test kümesi üzerinde modelin değerlendirilmesi
print("Accuracy:", knn.score(xtest1, ytest))

# Farklı komşu sayıları için doğruluk skorlarının hesaplanması
scorelite = []
for i in range(1, 30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(xtrain1, ytrain)
    scorelite.append(knn2.score(xtest1, ytest))

# Doğruluk skorlarının görselleştirilmesi
plt.plot(range(1, 30), scorelite)
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()


