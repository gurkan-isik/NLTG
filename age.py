import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. CC Verisinden Gelen GÜNCEL Parametreler
H0 = 69.97        # km/s/Mpc (Yeni değer)
w_eff = -1.115    # NLTG Infall Index (Yeni değer)
Omega_m = 0.315   # Madde Yoğunluğu (Sabit)
Omega_de = 1 - Omega_m

# H0'ı yıl^(-1) birimine çevirme
H0_yr = H0 * (3.1536e7 / 3.08567758e19)
H0_Gyr = H0_yr * 1e9

# ... (Kodun geri kalanı aynı kalacak, sadece yukarıdaki değerler değişti)


# 2. Friedmann Genişleme Oranı H(a) / H0
def E(a):
    return np.sqrt(Omega_m * a**(-3) + Omega_de * a**(-3 * (1 + w_eff)))

# 3. Evrenin Yaşını Hesaplama Fonksiyonu t(a)
def t_of_a(a_target):
    integrand = lambda a: 1.0 / (a * E(a))
    integral, _ = quad(integrand, 1e-10, a_target)
    return integral / H0_Gyr

# 4. İvmelenme Parametresi q(a)
def q(a):
    # q = -1 - (a/E) * (dE/da)
    # dE/da analitik türevi:
    term_m = -3 * Omega_m * a**(-4)
    term_de = -3 * (1 + w_eff) * Omega_de * a**(-3 * (1 + w_eff) - 1)
    dE_da = (0.5 / E(a)) * (term_m + term_de)
    return -1 - (a / E(a)) * dE_da

# Dizi oluşturma
a_vals = np.linspace(0.1, 1.8, 200)
t_vals = [t_of_a(a) for a in a_vals]
q_vals = [q(a) for a in a_vals]

# Günümüz yaşı (a=1)
t_today = t_of_a(1.0)

# 5. GÖRSELLEŞTİRME (Çift Eksenli Alt/Üst Grafik)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# --- ÜST GRAFİK: Scale Factor a(t) ---
ax1.plot(t_vals, a_vals, 'r-', lw=3, label=f'NLTG Scale Factor ($w_{{eff}}={w_eff}$)')
ax1.axvline(x=t_today, color='gray', linestyle='--', alpha=0.7)
ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

ax1.text(t_today - 0.3, 0.4, f'Present Day\n($t_0 \\approx {t_today:.2f}$ Gyr)', 
         rotation=90, color='dimgray', fontsize=11, va='center')
ax1.text(3, 0.4, '1. DECELERATION\n(Matter Dominated)', fontsize=11)
ax1.text(15, 1.4, '2. KINEMATIC INFALL\n(Accelerating towards $T_2$)', fontsize=11, color='darkred')

ax1.set_ylabel('Scale Factor $a(t)$', fontsize=12)
ax1.set_title('Cosmic Evolution and Kinematic Infall derived from CC Data', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.4)

# --- ALT GRAFİK: Deceleration Parameter q(t) ---
ax2.plot(t_vals, q_vals, 'navy', lw=2)
ax2.axhline(y=0, color='black', lw=1)
ax2.axvline(x=t_today, color='gray', linestyle='--', alpha=0.7)

# Taralı alanlar (Yavaşlama ve Hızlanma)
ax2.fill_between(t_vals, q_vals, 0, where=(np.array(q_vals) > 0), color='gray', alpha=0.2, label='Deceleration ($q > 0$)')
ax2.fill_between(t_vals, q_vals, 0, where=(np.array(q_vals) < 0), color='red', alpha=0.15, label='Acceleration ($q < 0$)')

ax2.set_xlabel('Cosmic Time (Billions of Years)', fontsize=12)
ax2.set_ylabel('Deceleration $q$', fontsize=12)
ax2.set_ylim(-1.5, 1.0)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('scale_factor_nltg.png', dpi=300)
plt.show()