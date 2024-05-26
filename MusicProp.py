import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.csv")

chroma_stft_mean = data['chroma_stft_mean']
chroma_stft_var = data['chroma_stft_var']
rms_mean = data['rms_mean']
rms_var = data['rms_var']
spectral_centroid_mean = data['spectral_centroid_mean']
spectral_centroid_var = data['spectral_centroid_var']
spectral_bandwidth_mean = data['spectral_bandwidth_mean']
spectral_bandwidth_var = data['spectral_bandwidth_var']
rolloff_mean = data['rolloff_mean']
rolloff_var = data['rolloff_var']
zero_crossing_rate_mean = data['zero_crossing_rate_mean']
zero_crossing_rate_var = data['zero_crossing_rate_var']
harmony_mean = data['harmony_mean']
harmony_var = data['harmony_var']
perceptr_mean = data['perceptr_mean']
perceptr_var = data['perceptr_var']
tempo = data['tempo']
mfcc1_mean = data['mfcc1_mean']
mfcc1_var = data['mfcc1_var']
mfcc2_mean = data['mfcc2_mean']
mfcc2_var = data['mfcc2_var']
mfcc3_mean = data['mfcc3_mean']
mfcc3_var = data['mfcc3_var']
mfcc4_mean = data['mfcc4_mean']
mfcc4_var = data['mfcc4_var']
mfcc5_mean = data['mfcc5_mean']
mfcc5_var = data['mfcc5_var']
mfcc6_mean = data['mfcc6_mean']
mfcc6_var = data['mfcc6_var']
mfcc7_mean = data['mfcc7_mean']
mfcc7_var = data['mfcc7_var']
mfcc8_mean = data['mfcc8_mean']
mfcc8_var = data['mfcc8_var']
mfcc9_mean = data['mfcc9_mean']
mfcc9_var = data['mfcc9_var']
mfcc10_mean = data['mfcc10_mean']
mfcc10_var = data['mfcc10_var']
mfcc11_mean = data['mfcc11_mean']
mfcc11_var = data['mfcc11_var']
mfcc12_mean = data['mfcc12_mean']
mfcc12_var = data['mfcc12_var']
mfcc13_mean = data['mfcc13_mean']
mfcc13_var = data['mfcc13_var']
mfcc14_mean = data['mfcc14_mean']
mfcc14_var = data['mfcc14_var']
mfcc15_mean = data['mfcc15_mean']
mfcc15_var = data['mfcc15_var']
mfcc16_mean = data['mfcc16_mean']
mfcc16_var = data['mfcc16_var']
mfcc17_mean = data['mfcc17_mean']
mfcc17_var = data['mfcc17_var']
mfcc18_mean = data['mfcc18_mean']
mfcc18_var = data['mfcc18_var']
mfcc19_mean = data['mfcc19_mean']
mfcc19_var = data['mfcc19_var']
mfcc20_mean = data['mfcc20_mean']
mfcc20_var = data['mfcc20_var']

"""
plt.scatter(chroma_stft_mean, chroma_stft_var, color="red")
plt.xlabel("chroma_stft_mean")
plt.ylabel("chroma_stft_var")
plt.legend()
plt.show()
"""

"""
plt.scatter(rms_mean, rms_var, color="blue")
plt.xlabel("rms_mean")
plt.ylabel("rms_var")
plt.legend()
#plt.show()
"""

"""
plt.scatter(spectral_centroid_mean, spectral_centroid_var, color="green", label="spectral_centroid")
plt.scatter(spectral_bandwidth_mean, spectral_bandwidth_var, color="orange", label="spectral_bandwidth")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
#plt.show()
"""
"""
plt.scatter(rolloff_mean, rolloff_var, color="purple", label="rolloff")
plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
#plt.show()
"""

"""
plt.scatter(zero_crossing_rate_mean, zero_crossing_rate_var, color="cyan", label="zero_crossing_rate")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()
"""

"""
plt.scatter(harmony_mean, harmony_var, color="yellow", label="harmony")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()
"""

"""
plt.scatter(perceptr_mean, perceptr_var, color="magenta", label="perceptr")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()
"""


"""
plt.scatter(tempo, tempo, color="black", label="tempo")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()
"""
"""
plt.scatter(mfcc1_mean, mfcc1_var, color="gray", label="mfcc1")
plt.scatter(mfcc2_mean, mfcc2_var, color="brown", label="mfcc2")
plt.scatter(mfcc3_mean, mfcc3_var, color="olive", label="mfcc3")
plt.scatter(mfcc4_mean, mfcc4_var, color="pink", label="mfcc4")
plt.scatter(mfcc5_mean, mfcc5_var, color="skyblue", label="mfcc5")
plt.scatter(mfcc6_mean, mfcc6_var, color="navy", label="mfcc6")
plt.scatter(mfcc7_mean, mfcc7_var, color="lime", label="mfcc7")
plt.scatter(mfcc8_mean, mfcc8_var, color="gold", label="mfcc8")
plt.scatter(mfcc9_mean, mfcc9_var, color="teal", label="mfcc9")
plt.scatter(mfcc10_mean, mfcc10_var, color="salmon", label="mfcc10")
plt.scatter(mfcc11_mean, mfcc11_var, color="indigo", label="mfcc11")
plt.scatter(mfcc12_mean, mfcc12_var, color="coral", label="mfcc12")
plt.scatter(mfcc13_mean, mfcc13_var, color="darkorange", label="mfcc13")
plt.scatter(mfcc14_mean, mfcc14_var, color="sienna", label="mfcc14")
plt.scatter(mfcc15_mean, mfcc15_var, color="darkslategray", label="mfcc15")
plt.scatter(mfcc16_mean, mfcc16_var, color="darkcyan", label="mfcc16")
plt.scatter(mfcc17_mean, mfcc17_var, color="peru", label="mfcc17")
plt.scatter(mfcc18_mean, mfcc18_var, color="darkred", label="mfcc18")
plt.scatter(mfcc19_mean, mfcc19_var, color="cadetblue", label="mfcc19")
plt.scatter(mfcc20_mean, mfcc20_var, color="mediumorchid", label="mfcc20")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()
"""

plt.scatter(chroma_stft_mean, chroma_stft_var, color="red", label="chroma_stft")
plt.scatter(rms_mean, rms_var, color="blue", label="rms")
plt.scatter(spectral_centroid_mean, spectral_centroid_var, color="green", label="spectral_centroid")
plt.scatter(spectral_bandwidth_mean, spectral_bandwidth_var, color="orange", label="spectral_bandwidth")
plt.scatter(rolloff_mean, rolloff_var, color="purple", label="rolloff")
plt.scatter(zero_crossing_rate_mean, zero_crossing_rate_var, color="cyan", label="zero_crossing_rate")
plt.scatter(harmony_mean, harmony_var, color="yellow", label="harmony")
plt.scatter(perceptr_mean, perceptr_var, color="magenta", label="perceptr")
plt.scatter(tempo, tempo, color="black", label="tempo")
plt.scatter(mfcc1_mean, mfcc1_var, color="gray", label="mfcc1")
plt.scatter(mfcc2_mean, mfcc2_var, color="brown", label="mfcc2")
plt.scatter(mfcc3_mean, mfcc3_var, color="olive", label="mfcc3")
plt.scatter(mfcc4_mean, mfcc4_var, color="pink", label="mfcc4")
plt.scatter(mfcc5_mean, mfcc5_var, color="skyblue", label="mfcc5")
plt.scatter(mfcc6_mean, mfcc6_var, color="navy", label="mfcc6")
plt.scatter(mfcc7_mean, mfcc7_var, color="lime", label="mfcc7")
plt.scatter(mfcc8_mean, mfcc8_var, color="gold", label="mfcc8")
plt.scatter(mfcc9_mean, mfcc9_var, color="teal", label="mfcc9")
plt.scatter(mfcc10_mean, mfcc10_var, color="salmon", label="mfcc10")
plt.scatter(mfcc11_mean, mfcc11_var, color="indigo", label="mfcc11")
plt.scatter(mfcc12_mean, mfcc12_var, color="coral", label="mfcc12")
plt.scatter(mfcc13_mean, mfcc13_var, color="darkorange", label="mfcc13")
plt.scatter(mfcc14_mean, mfcc14_var, color="sienna", label="mfcc14")
plt.scatter(mfcc15_mean, mfcc15_var, color="darkslategray", label="mfcc15")
plt.scatter(mfcc16_mean, mfcc16_var, color="darkcyan", label="mfcc16")
plt.scatter(mfcc17_mean, mfcc17_var, color="peru", label="mfcc17")
plt.scatter(mfcc18_mean, mfcc18_var, color="darkred", label="mfcc18")
plt.scatter(mfcc19_mean, mfcc19_var, color="cadetblue", label="mfcc19")
plt.scatter(mfcc20_mean, mfcc20_var, color="mediumorchid", label="mfcc20")

plt.xlabel("Mean")
plt.ylabel("Variance")
plt.legend()
plt.show()










