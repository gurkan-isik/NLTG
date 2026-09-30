import numpy as np
from scipy.optimize import minimize

x  = np.log10(Mb)                    # baryonik kütle [Msun]
y  = np.log10(Vf)                    # km/s
sy = sigVf / (Vf * np.log(10))       # log V hatası

def nll(p, alpha=None):
    if alpha is None: b, a, lsi = p          # b = log10(k)
    else:             b, lsi = p; a = alpha
    s2 = sy**2 + np.exp(2*lsi)               # sigma_int = exp(lsi) [dex]
    r  = y - (b + a*x)
    return 0.5*np.sum(r**2/s2 + np.log(2*np.pi*s2))

free  = minimize(nll, [np.log10(0.39), 0.25, np.log(0.03)], method='Nelder-Mead')
fixed = minimize(lambda p: nll(p, 0.25), [np.log10(0.39), np.log(0.03)], method='Nelder-Mead')
b, a, lsi = free.x
print("alpha =", a, " sigma_int [dex] =", np.exp(lsi), " k =", 10**b)
print("dAIC (free - fixed slope) =", (2*free.fun + 2*3) - (2*fixed.fun + 2*2))