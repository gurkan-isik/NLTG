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