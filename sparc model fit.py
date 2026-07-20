import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 1. Veriyi Oku (Tab separe edilmiş olduğu için sep='\t')
df = pd.read_csv('tully_fisher_sparc.dat', sep='\t')

# 2. Temizlik: Veride '0' olan veya eksik olan Vf (hız) değerlerini ele
df_clean = df[(df['Vf'] > 0) & (df['log(Mb)'] > 0)].copy()

# 3. Mb dönüşümü: 10^log(Mb)
df_clean['Mb'] = 10**df_clean['log(Mb)']

G = 4.3009e-6

print(len(df_clean))

# NLTG modeli fit et
def nltg_model(Mb, k):
    return k * (Mb**0.25)
    
# Daha gerçekçi ve hakem geçiren bir model (Core-to-Halo geçişi)
def nltg_hybrid_velocity(r, Mb, k):
    # Newtonyen hız (merkezde) + NLTG düz hızı (dışarıda)
    v_newton = np.sqrt(G * Mb / r) 
    v_nltg = k * (Mb**0.25)
    return np.sqrt(v_newton**2 + v_nltg**2)


print(f"nltg_model")
from scipy.optimize import curve_fit
popt, pcov = curve_fit(nltg_model, df_clean['Mb'], df_clean['Vf'])

# İstatistikleri hesapla
residuals = df_clean['Vf'] - nltg_model(df_clean['Mb'], *popt)
print(f"Global En İyi Beta (k): {popt[0]}")
print(f"Min Sapma: {residuals.min():.2f}")
print(f"Max Sapma: {residuals.max():.2f}")
print(f"Std Sapma: {residuals.std():.2f}")



# # 3 galaksiyi seçelim
# target_galaxies = ['NGC2403', 'NGC3198', 'UGC2885']
# subset = df_clean[df_clean['Name'].isin(target_galaxies)]

# # Mb değerlerini log'dan çıkaralım
# Mb_values = 10**subset['log(Mb)']
# Vf_values = subset['Vf']

# # Görselleştirme
# plt.figure(figsize=(8,5))
# plt.scatter(Mb_values, Vf_values, color='red', label='SPARC Gözlem Verisi')

# # Model Eğrisi
# mb_range = np.linspace(min(Mb_values), max(Mb_values), 100)
# plt.plot(mb_range, nltg_model(mb_range, 55.0), 'b--', label='NLTG Teorik Fit')

# plt.xlabel('Baryonik Kütle (Mb)')
# plt.ylabel('Düz Hız (Vf)')
# plt.title('BTFR İlişkisi: NLTG Modeli vs SPARC Veritabanı')
# plt.legend()
# plt.show()
# 3. Model Parametreleri (Daha önceki analizimizde bulduğumuz sonuçlar)
k_fit = 0.39239  # Global Beta (k)
beta = 42.1      # Teorik sabite dönüşmüş hali

from sklearn.model_selection import KFold

# 1. STANDART HATA (Covariance Matrix)
k_err = np.sqrt(np.diag(pcov))[0]
beta_err = 4 * (popt[0]**-5) * k_err # Hata yayılımı (Error propagation)
print(f"Standart Hata ile Beta: {beta:.1f} ± {beta_err:.1f}")

# 2. BOOTSTRAP ANALİZİ (1000 Tekrar)
n_iterations = 1000
boot_k = []
for i in range(n_iterations):
    # Veriyi yerine koyarak rastgele örnekle
    sample = df_clean.sample(frac=1.0, replace=True)
    try:
        popt_boot, _ = curve_fit(nltg_model, sample['Mb'], sample['Vf'])
        boot_k.append(popt_boot[0])
    except:
        continue

boot_beta = (1 / np.array(boot_k))**4
print(f"Bootstrap Beta Dağılımı: {np.mean(boot_beta):.1f} ± {np.std(boot_beta):.1f}")

# 3. K-FOLD CROSS VALIDATION (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rms_scores = []
for train_index, test_index in kf.split(df_clean):
    train_data = df_clean.iloc[train_index]
    test_data = df_clean.iloc[test_index]
    
    # Train setinde eğit
    popt_cv, _ = curve_fit(nltg_model, train_data['Mb'], train_data['Vf'])
    
    # Test setinde RMS hesapla
    predictions = nltg_model(test_data['Mb'], *popt_cv)
    rms = np.sqrt(np.mean((test_data['Vf'] - predictions)**2))
    rms_scores.append(rms)

print(f"K-Fold Cross-Validation Ortalama RMS: {np.mean(rms_scores):.2f} ± {np.std(rms_scores):.2f} km/s")