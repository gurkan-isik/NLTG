import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Veriyi Oku
df = pd.read_csv('tully_fisher_sparc.dat', sep='\t')

# 2. Veri Temizliği (Gözlem kalitesi Q=3 olanları ve i < 30 olanları eliyoruz)
# Eğer verinizde 'Q' ve 'i' (inclination) sütunları varsa aşağıdaki satırı aktif edin:
# df_clean = df[(df['Vf'] > 0) & (df['log(Mb)'] > 0) & (df['Q'] < 3) & (df['i'] >= 30)].copy()

# Eğer Q ve i sütunları yoksa, eski temizliğinizi kullanıyoruz:
df_clean = df[(df['Vf'] > 0) & (df['log(Mb)'] > 0)].copy()
df_clean['Mb'] = 10**df_clean['log(Mb)']

# 3. Modelleri Tanımla

# Model A: Sizin NLTG Teorik Modeliniz (Üs 0.25'e yani 1/4'e sabitlenmiş)
def theoretical_model(Mb, k):
    return k * (Mb**0.25)

# Model B: Hakemin İstediği Serbest Eğimli (Free-Slope) Model
def free_slope_model(Mb, k, alpha):
    return k * (Mb**alpha)

# 4. Eğri Uydurma (Curve Fitting)

# A) Teorik Fit (Sadece k'yı bulur)
popt_theo, pcov_theo = curve_fit(theoretical_model, df_clean['Mb'], df_clean['Vf'])
k_theo = popt_theo[0]

# B) Serbest Fit (Hem k'yı hem alpha'yı bulur)
# Başlangıç tahmini olarak k=0.39, alpha=0.25 veriyoruz
popt_free, pcov_free = curve_fit(free_slope_model, df_clean['Mb'], df_clean['Vf'], p0=[0.39, 0.25])
k_free = popt_free[0]
alpha_free = popt_free[1]
alpha_err = np.sqrt(pcov_free[1, 1]) # Alpha'nın standart hata payı

# Sonuçları Ekrana Bas (Makaleye yazılacak değerler)
print("-" * 50)
print(f"Teorik Model k değeri: {k_theo:.4f}")
print(f"SERBEST FIT SONUÇLARI (Makaleye Yazılacak):")
print(f"Vf = k * Mb^alpha formatında best-fit alpha = {alpha_free:.4f} ± {alpha_err:.4f}")
print(f"Mb = A * Vf^beta (Ters) formatında best-fit beta = {1/alpha_free:.2f} ± {(alpha_err / (alpha_free**2)):.2f}")
print("-" * 50)

# 5. Görselleştirme (Grafik)
plt.figure(figsize=(9, 6))

# Gözlem noktaları
plt.errorbar(df_clean['Mb'], df_clean['Vf'], yerr=df_clean['e_Vf'], 
             fmt='o', color='black', ecolor='gray', alpha=0.5, markersize=5, 
             label=f'SPARC Data ({len(df_clean)} Galaxies)')

# X ekseni aralığı
mb_range = np.logspace(np.log10(df_clean['Mb'].min()), np.log10(df_clean['Mb'].max()), 100)

# Teorik Çizgi (Kesikli Mavi)
plt.plot(mb_range, theoretical_model(mb_range, k_theo), 'b--', linewidth=2, 
         label=f'NLTG Prediction ($V_f \propto M_b^{{0.25}}$)')

# Serbest Fit Çizgisi (Düz Kırmızı)
plt.plot(mb_range, free_slope_model(mb_range, k_free, alpha_free), 'r-', linewidth=2.5, 
         label=f'Free-Slope Best Fit ($V_f \propto M_b^{{{alpha_free:.3f}}}$)')

# Eksen Ayarları
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Baryonic Mass $M_b$ [$M_\odot$]', fontsize=13)
plt.ylabel('Flat Rotation Velocity $V_f$ [km/s]', fontsize=13)
plt.title('Baryonic Tully-Fisher Relation: Theoretical vs Free-Slope Fit', fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('sparc_free_fit_comparison.png', dpi=300)
plt.show()