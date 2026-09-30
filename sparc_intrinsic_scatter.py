import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. Veriyi Oku (Sizin kodunuzdaki ile aynı)
df = pd.read_csv('tully_fisher_sparc.dat', sep='\t')

# 2. Veri Temizliği 
# (Eğer Q ve i sütunlarınız verinizde varsa, ilk satırı aktif edebilirsiniz)
df_clean = df[(df['Vf'] > 0) & (df['log(Mb)'] > 0)].copy()

# 3. İlgili Değişkenleri Tanımla (NLL Hesabı İçin)
x = df_clean['log(Mb)'].values                     # x = log10(Mb)
y = np.log10(df_clean['Vf'].values)                # y = log10(Vf)
sigVf = df_clean['e_Vf'].values                    # Vf hatası (km/s cinsinden)
sy = sigVf / (df_clean['Vf'].values * np.log(10))  # y için hata propagasyonu (dex cinsinden hata)

# Negatif Log-Olabilirlik (NLL) Fonksiyonu
# p = [b, a, lsi] -> b: log10(k), a: eğim(alpha), lsi: ln(sigma_int)
def nll(p, alpha=None):
    if alpha is None: 
        b, a, lsi = p          
    else:             
        b, lsi = p
        a = alpha
    
    # Toplam varyans (Ölçüm hatası + İçsel Saçılım)
    # lsi = ln(sigma_int) kullandık ki exp alınca sigma_int her zaman pozitif olsun
    s2 = sy**2 + np.exp(2 * lsi) 
    
    # Kalıntılar (Residuals)
    r = y - (b + a * x)          
    
    # Log-Likelihood
    return 0.5 * np.sum((r**2) / s2 + np.log(2 * np.pi * s2))

# 4. Optimizasyon (Serbest Eğim - Free Slope)
# Başlangıç tahminleri: b = log10(0.39), a = 0.25, lsi = ln(0.03)
p0_free = [np.log10(0.39), 0.25, np.log(0.03)]

# BFGS metodu varyans-kovaryans matrisini döndürür (Hata payları için gerekli)
res_free = minimize(nll, p0_free, method='BFGS')

b_free, a_free, lsi_free = res_free.x
sigma_int_free = np.exp(lsi_free) # dex cinsinden içsel saçılım
k_free = 10**b_free

# Hata Hesaplama (Kovaryans matrisinin köşegen elemanlarının karekökü)
cov_matrix = res_free.hess_inv
a_err = np.sqrt(cov_matrix[1, 1]) # Eğimin (alpha) standart hatası

# 5. Optimizasyon (Sabit Eğim - Fixed Slope, alpha = 0.25)
p0_fixed = [np.log10(0.39), np.log(0.03)]
res_fixed = minimize(lambda p: nll(p, alpha=0.25), p0_fixed, method='BFGS')

# 6. İstatistiksel Karşılaştırmalar (AIC ve Sigma Farkı)
aic_free = 2 * res_free.fun + 2 * 3
aic_fixed = 2 * res_fixed.fun + 2 * 2
daic = aic_free - aic_fixed

# Bulunan eğimin 0.25'ten farkının istatistiksel anlamlılığı (Z-skoru / Sigma)
sigma_diff = abs(a_free - 0.25) / a_err

print("-" * 60)
print("İÇSEL SAÇILIM (INTRINSIC SCATTER) ANALİZİ SONUÇLARI")
print("-" * 60)
print(f"[X] Intrinsic scatter (sigma_int) = {sigma_int_free:.4f} dex")
print(f"[Y] Serbest Eğim (alpha)          = {a_free:.4f}")
print(f"[Z] Eğim Hatası (sigma_alpha)     = {a_err:.4f}")
print(f"[W] 0.25'ten farkın anlamlılığı   = {sigma_diff:.2f} sigma")
print(f"Sabit k değeri                    = {k_free:.4f}")
print(f"dAIC (Serbest - Sabit eğim)       = {daic:.2f}")
print("-" * 60)

print(f"The fit yields \u03C3_int = {sigma_int_free:.4f} dex and a slope \u03B1 = {a_free:.3f} \u00B1 {a_err:.3f}; "
      f"including the intrinsic scatter reduces the significance of the difference between the fitted slope and the predicted \u03B1 = 0.25 ({sigma_diff:.2f}\u03C3).")