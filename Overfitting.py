import librosa
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Örnek müziği yükleme ve özelliklerini çıkarma
def extract_features(file_path):
    # Müziği yükleme
    y, sr = librosa.load(file_path, sr=None)
    features = []
    # Özellikleri çıkarma
    # Özellikleri çıkarma
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))

    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))

    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))
    # Özellikleri çıkarma ve ortalama/varyans hesaplama

    #perceptr ve harmony li,brosa kütüphensinde olmamasından çıkatıldı

    #tempo = np.mean(librosa.beat.tempo(y=y, sr=sr))

# MFCC özellikleri için
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_means = np.mean(mfccs, axis=1)  # Her bir MFCC'nin ortalama değeri
    mfcc_vars = np.var(mfccs, axis=1)    # Her bir MFCC'nin varyans değeri

# Her bir MFCC için ortalama ve varyansı ayırma
    mfcc1_mean = mfcc_means[0]
    mfcc1_var = mfcc_vars[0]

    mfcc2_mean = mfcc_means[1]
    mfcc2_var = mfcc_vars[1]
    
    mfcc3_mean = mfcc_means[2]
    mfcc3_var = mfcc_vars[2]

    mfcc4_mean = mfcc_means[3]
    mfcc4_var = mfcc_vars[3]

    mfcc5_mean = mfcc_means[4]
    mfcc5_var = mfcc_vars[4]

    mfcc6_mean = mfcc_means[5]
    mfcc6_var = mfcc_vars[5]

    mfcc7_mean = mfcc_means[6]
    mfcc7_var = mfcc_vars[6]

    mfcc8_mean = mfcc_means[7]
    mfcc8_var = mfcc_vars[7]

    mfcc9_mean = mfcc_means[8]
    mfcc9_var = mfcc_vars[8]

    mfcc10_mean = mfcc_means[9]
    mfcc10_var = mfcc_vars[9]

    mfcc11_mean = mfcc_means[10]
    mfcc11_var = mfcc_vars[10]

    mfcc12_mean = mfcc_means[11]
    mfcc12_var = mfcc_vars[11]

    mfcc13_mean = mfcc_means[12]
    mfcc13_var = mfcc_vars[12]

    mfcc14_mean = mfcc_means[13]
    mfcc14_var = mfcc_vars[13]

    mfcc15_mean = mfcc_means[14]
    mfcc15_var = mfcc_vars[14]

    mfcc16_mean = mfcc_means[15]
    mfcc16_var = mfcc_vars[15]

    mfcc17_mean = mfcc_means[16]
    mfcc17_var = mfcc_vars[16]

    mfcc18_mean = mfcc_means[17]
    mfcc18_var = mfcc_vars[17]

    mfcc19_mean = mfcc_means[18]
    mfcc19_var = mfcc_vars[18]

    mfcc20_mean = mfcc_means[19]
    mfcc20_var = mfcc_vars[19]
# Bu şekilde devam ederek diğer MFCC'lerin ortalama ve varyanslarını elde edebilirsiniz

# Diğer özelliklerin de benzer şekilde çıkarılması gerekecek

    
    # Özellikleri birleştir
    # features listesine özelliklerin hepsini ekleyelim
    features.extend([
    chroma_stft_mean, chroma_stft_var,
    rms_mean, rms_var,
    spectral_centroid_mean, spectral_centroid_var,
    spectral_bandwidth_mean, spectral_bandwidth_var,
    rolloff_mean, rolloff_var,
    zero_crossing_rate_mean, zero_crossing_rate_var,
    #tempo
    ])

# MFCC özellikleri
    mfcc_means = [mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean, mfcc6_mean, mfcc7_mean, mfcc8_mean, mfcc9_mean, mfcc10_mean, mfcc11_mean, mfcc12_mean, mfcc13_mean, mfcc14_mean, mfcc15_mean, mfcc16_mean, mfcc17_mean, mfcc18_mean, mfcc19_mean, mfcc20_mean]
    mfcc_vars = [mfcc1_var, mfcc2_var, mfcc3_var, mfcc4_var, mfcc5_var, mfcc6_var, mfcc7_var, mfcc8_var, mfcc9_var, mfcc10_var, mfcc11_var, mfcc12_var, mfcc13_var, mfcc14_var, mfcc15_var, mfcc16_var, mfcc17_var, mfcc18_var, mfcc19_var, mfcc20_var]

    features.extend(mfcc_means)
    features.extend(mfcc_vars)

    
    return features

# Eğitilmiş modeli yükleme veya oluşturma
def train_model():
    # GTZAN veri setini yükleme (Önceden hazırlanmış bir örneği kullanıyoruz)
    data_30_sec = pd.read_csv("C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.csv")

    # Özellikleri ve etiketleri ayırma
    X = data_30_sec.drop(['filename','perceptr_var','perceptr_mean','harmony_var','harmony_mean','length','tempo','label'], axis=1)

    y = data_30_sec['label']

    # Verileri normalleştirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KNN modelini oluşturma ve eğitme
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)

    return knn

# Tahmin yapma
def predict_genre(file_path, model):
    # Özellikleri çıkarma
    features = extract_features(file_path)
    
    # Özellikleri normalleştirme
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])

    # Tahmin yapma
    predicted_genre = model.predict(features_scaled)
    
    return predicted_genre[0]

# Eğitilmiş modeli oluşturma
model = train_model()

# Örnek müziği belirt
example_music_file = r"C:/Users/efefa/Desktop/Python-Proje/Data/genres_original/blues/blues.00031.wav"


# Tahmini müzik türünü al
predicted_genre = predict_genre(example_music_file, model)

# Tahmini müzik türünü yazdır
print("Tahmin edilen müzik türü:", predicted_genre)
