import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Veriyi Oku
df = pd.read_csv('tully_fisher_sparc.dat', sep='\t')

# 2. Veri Temizliği ve Dönüşümü
df_clean = df[(df['Vf'] > 0) & (df['log(Mb)'] > 0)].copy()
df_clean['Mb'] = 10**df_clean['log(Mb)']

# 3. Model Parametreleri (Daha önceki analizimizde bulduğumuz sonuçlar)
k_fit = 0.39239  # Global Beta (k)
beta = 42.1      # Teorik sabite dönüşmüş hali

# NLTG Modeli: V = k * Mb^0.25
def nltg_model(Mb, k):
    return k * (Mb**0.25)

# 4. Görselleştirme
plt.figure(figsize=(9, 6))

# Gözlem noktaları ve Hata Çubukları (e_Vf sütununu kullanıyoruz)
plt.errorbar(df_clean['Mb'], df_clean['Vf'], yerr=df_clean['e_Vf'], 
             fmt='o', color='black', ecolor='gray', alpha=0.6, markersize=5, 
             label='SPARC Data (119 Galaxies)')

# Fit Eğrisi (Kırmızı ve belirgin)
mb_range = np.logspace(np.log10(df_clean['Mb'].min()), np.log10(df_clean['Mb'].max()), 100)
plt.plot(mb_range, nltg_model(mb_range, k_fit), 'r-', linewidth=2.5, 
         label=f'NLTG Theoretical Fit ($V_f \propto M_b^{{0.25}}$)\n$k={k_fit:.3f}, \\beta={beta:.1f}$')

# Eksen Ayarları (Log-Log Ölçek)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Baryonic Mass $M_b$ [$M_\odot$]', fontsize=13)
plt.ylabel('Flat Rotation Velocity $V_f$ [km/s]', fontsize=13)
plt.title('Baryonic Tully-Fisher Relation via NLTG Worldtube Potential', fontsize=14)

# Grid ve Legend
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc='upper left', fontsize=11)

# Makale için yüksek çözünürlüklü kaydet
plt.tight_layout()
plt.savefig('sparc_nltg_fit.png', dpi=300)
plt.show()



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