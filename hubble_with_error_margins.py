import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. GERÇEK KOZMİK KRONOMETRE VERİLERİ (Moresco et al., Stern et al. derlemesi)
# Sütunlar: Redshift (z), Hubble Parametresi H(z) [km/s/Mpc], Hata Payı (err_H)
cc_data = np.array([
    [0.070, 69.0, 19.6], [0.090, 69.0, 12.0], [0.120, 68.6, 26.2],
    [0.170, 83.0, 8.0],  [0.179, 75.0, 4.0],  [0.199, 75.0, 5.0],
    [0.200, 72.9, 29.6], [0.270, 77.0, 14.0], [0.280, 88.8, 36.6],
    [0.352, 83.0, 14.0], [0.380, 83.0, 13.5], [0.400, 95.0, 17.0],
    [0.425, 87.1, 11.2], [0.445, 89.2, 8.4],  [0.470, 89.0, 50.0],
    [0.478, 80.9, 9.0],  [0.480, 97.0, 62.0], [0.593, 104.0, 13.0],
    [0.680, 92.0, 8.0],  [0.781, 105.0, 12.0], [0.875, 125.0, 17.0],
    [0.880, 90.0, 40.0], [0.900, 117.0, 23.0], [1.037, 154.0, 20.0],
    [1.300, 168.0, 17.0], [1.363, 160.0, 33.6], [1.430, 177.0, 18.0],
    [1.530, 140.0, 14.0], [1.750, 202.0, 40.0], [1.965, 186.5, 50.4]
])

z_obs = cc_data[:, 0]
H_obs = cc_data[:, 1]
err_H = cc_data[:, 2]

# 2. TEORİK MODELLER
# Madde yoğunluğu güncel literatür değeri (yaklaşık 0.315)
Omega_m = 0.315

def nltg_hz(z, H0, w_eff):
    """
    NLTG Kinematik Düşüş Modeli (Makalenizdeki Denklem 16-18'in Friedmann açılımı)
    w_eff: NLTG'nin yarattığı efektif durum denklemi
    """
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m) * (1 + z)**(3 * (1 + w_eff)))

def lcdm_hz_fit(z, H0, Om_m):
    """Standart Lambda-CDM Modeli (2 Serbest Parametreli Fit İçin)"""
    return H0 * np.sqrt(Om_m * (1 + z)**3 + (1 - Om_m))

def lcdm_hz_fixed(z, H0):
    """Standart Lambda-CDM Modeli (Grafikte Karşılaştırma İçin w = -1, Om_m sabit)"""
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

# 3. İSTATİSTİKSEL FİT İŞLEMİ VE KOVARYANS (HATA) ANALİZİ
# NLTG modelini hem H0 hem de w_eff için serbest bırakarak fit ediyoruz.
# absolute_sigma=True parametresi, hata paylarının (pcov) doğru hesaplanmasını sağlar.
popt_nltg, pcov_nltg = curve_fit(nltg_hz, z_obs, H_obs, sigma=err_H, p0=[73.0, -1.2], absolute_sigma=True)
H0_fit, w_fit = popt_nltg
h0_err = np.sqrt(pcov_nltg[0, 0])
w_err = np.sqrt(pcov_nltg[1, 1])

# Lambda-CDM modelini fit ediyoruz
popt_lcdm, pcov_lcdm = curve_fit(lcdm_hz_fit, z_obs, H_obs, sigma=err_H, p0=[70.0, 0.3], absolute_sigma=True)
H0_lcdm_fit, Om_m_fit = popt_lcdm
h0_lcdm_err = np.sqrt(pcov_lcdm[0, 0])
om_lcdm_err = np.sqrt(pcov_lcdm[1, 1])

# 4. GÖRSELLEŞTİRME
plt.figure(figsize=(10, 6))
plt.errorbar(z_obs, H_obs, yerr=err_H, fmt='o', color='black', label='Cosmic Chronometers Data', alpha=0.7)

z_range = np.linspace(0, 2.0, 100)

# NLTG Eğrisi (Fit edilmiş parametrelerle)
plt.plot(z_range, nltg_hz(z_range, H0_fit, w_fit), 'r-', linewidth=2.5, 
         label=f'NLTG Fit ($H_0$={H0_fit:.2f}, $w_{{eff}}$={w_fit:.3f})')

# Karşılaştırma için Standart ACDM (Fit sonucundaki H0 ile)
plt.plot(z_range, lcdm_hz_fixed(z_range, 68.47), 'b--', linewidth=2, 
         label='Standard $\Lambda$CDM ($H_0$=68.47, $w$=-1)')

plt.xlabel('Redshift ($z$)', fontsize=12)
plt.ylabel('Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=12)
plt.title('Statistical Comparison: NLTG vs $\Lambda$CDM', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Grafiği kaydet
plt.savefig('cosmicfit.png', dpi=300)
plt.show()

# 5. İSTATİSTİKSEL SONUÇLAR VE AIC/BIC KARŞILAŞTIRMASI
chi2_nltg = np.sum(((H_obs - nltg_hz(z_obs, H0_fit, w_fit)) / err_H)**2)
chi2_lcdm = np.sum(((H_obs - lcdm_hz_fit(z_obs, H0_lcdm_fit, Om_m_fit)) / err_H)**2)
dof = len(z_obs) - 2 # 2 serbest parametre
N = len(z_obs)       # Veri sayısı (30)
k = 2                # Serbest parametre sayısı

# Metrikler
aic_nltg = chi2_nltg + 2 * k
bic_nltg = chi2_nltg + k * np.log(N)
aic_lcdm = chi2_lcdm + 2 * k
bic_lcdm = chi2_lcdm + k * np.log(N)
delta_aic = aic_nltg - aic_lcdm
delta_bic = bic_nltg - bic_lcdm

print("--- NLTG FIT SONUÇLARI ---")
print(f"En İyi H0: {H0_fit:.2f} ± {h0_err:.2f} km/s/Mpc")
print(f"En İyi Efektif w (Infall Index): {w_fit:.3f} ± {w_err:.3f}")
print(f"Ki-kare / DoF: {chi2_nltg/dof:.2f}")

print("\n--- STANDART LAMBDA-CDM FIT SONUÇLARI ---")
print(f"En İyi H0: {H0_lcdm_fit:.2f} ± {h0_lcdm_err:.2f} km/s/Mpc")
print(f"En İyi Omega_m: {Om_m_fit:.3f} ± {om_lcdm_err:.3f}")
print(f"Ki-kare / DoF: {chi2_lcdm/dof:.2f}")

print("\n--- MODEL KARŞILAŞTIRMA METRİKLERİ (TABLO 4 İÇİN) ---")
print(f"NLTG Chi2: {chi2_nltg:.2f} | AIC: {aic_nltg:.2f} | BIC: {bic_nltg:.2f}")
print(f"LCDM Chi2: {chi2_lcdm:.2f} | AIC: {aic_lcdm:.2f} | BIC: {bic_lcdm:.2f}")
print(f"Delta AIC: {delta_aic:.2f} (Eksi değer NLTG lehine)")
print(f"Delta BIC: {delta_bic:.2f} (Eksi değer NLTG lehine)")

# 6. ALT ÖRNEKLEME VE KARARLILIK TESTLERİ
print("\n--- BAŞLANGIÇ DEĞERİ KARARLILIK TESTİ ---")
for p0 in [[65.0, -0.5], [75.0, -2.0], [70.0, -1.0]]:
    popt_test, _ = curve_fit(nltg_hz, z_obs, H_obs, sigma=err_H, p0=p0, absolute_sigma=True)
    print(f"Başlangıç {p0} -> Sonuç: H0={popt_test[0]:.2f}, w_eff={popt_test[1]:.3f}")

n_sub_iterations = 1000
delta_aics = []
n_subsample = int(0.8 * len(z_obs)) 

for i in range(n_sub_iterations):
    idx = np.random.choice(len(z_obs), size=n_subsample, replace=False)
    z_sub, H_sub, err_sub = z_obs[idx], H_obs[idx], err_H[idx]
    try:
        popt_nltg_sub, _ = curve_fit(nltg_hz, z_sub, H_sub, sigma=err_sub, p0=[70.0, -1.2], absolute_sigma=True)
        chi2_nltg_sub = np.sum(((H_sub - nltg_hz(z_sub, *popt_nltg_sub)) / err_sub)**2)
        aic_nltg_sub = chi2_nltg_sub + 2 * k
        
        popt_lcdm_sub, _ = curve_fit(lcdm_hz_fit, z_sub, H_sub, sigma=err_sub, p0=[70.0, 0.3], absolute_sigma=True)
        chi2_lcdm_sub = np.sum(((H_sub - lcdm_hz_fit(z_sub, *popt_lcdm_sub)) / err_sub)**2)
        aic_lcdm_sub = chi2_lcdm_sub + 2 * k
        
        delta_aics.append(aic_nltg_sub - aic_lcdm_sub)
    except:
        continue

print(f"\n--- ALT ÖRNEKLEME TESTİ (1000 Tekrar, %80 Veri) ---")
print(f"Ortalama Delta AIC: {np.mean(delta_aics):.3f} ± {np.std(delta_aics):.3f}")

chi2_planck = np.sum(((H_obs - lcdm_hz_fit(z_obs, 67.4, 0.315)) / err_H)**2)
aic_planck = chi2_planck + 2 * k
print(f"\nPlanck 2018 LCDM (H0=67.4) karşı Delta AIC: {aic_nltg - aic_planck:.2f} (Eksi değer NLTG lehine)")