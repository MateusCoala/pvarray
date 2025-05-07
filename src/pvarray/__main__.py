import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar, differential_evolution

# === Constantes físicas ===
q = 1.602e-19
k = 1.381e-23
Tc = 298.15
Gamma = 1.2
Ncs = 72

# === Dados (V, I)
data = np.array([
    [0.0, 9.20],
    [37.83, 8.72],
    [45.19, 0.0]
])

# === Modelo
def pv_current_model(params, V):
    Iph, Io, Rs, Rsh = params
    I_est = []
    for v in V:
        def implicit_eq(I):
            try:
                expo = q * (v + I * Rs) / (Ncs * Gamma * k * Tc)
                return Iph - Io * (np.exp(expo) - 1) - (v + I * Rs) / Rsh - I
            except:
                return np.inf
        try:
            a, b = 0.0, Iph + 2
            fa, fb = implicit_eq(a), implicit_eq(b)
            if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
                I_est.append(np.nan)
                continue
            sol = root_scalar(implicit_eq, method="brentq", bracket=[a, b])
            I_est.append(sol.root if sol.converged else np.nan)
        except:
            I_est.append(np.nan)
    return np.array(I_est)

# === Função objetivo
def func_objetivo(params):
    V = data[:, 0]
    I_real = data[:, 1]
    I_pred = pv_current_model(params, V)
    if np.any(np.isnan(I_pred)):
        return 1e6
    return np.mean((I_real - I_pred) ** 2)

# === Limites
limites = [(8, 10), (1e-12, 1e-6), (0.01, 2.0), (10, 10000)]

# === Otimização global
print(" Otimizando com Differential Evolution...")
resultado = differential_evolution(
    func_objetivo,
    bounds=limites,
    maxiter=3000,
    popsize=25,
    tol=1e-8,
    polish=True
)
params_otimizados = resultado.x

# === Refinamento local
print("\n Refinando com L-BFGS-B...\n")
refinamento = minimize(
    func_objetivo,
    params_otimizados,
    bounds=limites,
    method='L-BFGS-B',
    options={'maxiter': 5000, 'ftol': 1e-12}
)
params_finais = refinamento.x
IphRef, IoRef, Rs, Rsh = params_finais

# === Impressão
print("\n=== RESULTADO FINAL ===")
print(f"IphRef = {IphRef:.5f} A")
print(f"IoRef  = {IoRef:.5e} A")
print(f"Rs     = {Rs:.5f} Ω")
print(f"Rsh    = {Rsh:.2f} Ω")

# === Erros ponto a ponto
V_real = data[:, 0]
I_real = data[:, 1]
I_pred = pv_current_model(params_finais, V_real)

print("\n--- Erros ponto a ponto ---")
total_error = 0
for v, i_real, i_pred in zip(V_real, I_real, I_pred):
    if np.isnan(i_pred):
        print(f"V = {v:.2f} V | I_real = {i_real:.2f} A | I_pred = nan A | Erro = --")
        continue
    erro_abs = abs(i_real - i_pred)
    erro_perc = 100 * erro_abs / max(i_real, 1e-6)
    total_error += erro_perc
    print(f"V = {v:.2f} V | I_real = {i_real:.2f} A | I_pred = {i_pred:.2f} A | Erro = {erro_perc:.2f}%")

erro_medio = total_error / len([i for i in I_pred if not np.isnan(i)])
print(f"\n[✓] Erro percentual médio: {erro_medio:.2f}%")

# === Plot
V_plot = np.linspace(0, 46, 200)
I_model = pv_current_model(params_finais, V_plot)

plt.figure(figsize=(8, 5))
plt.plot(V_plot, I_model, label="Modelo ajustado", lw=2)
plt.scatter(data[:, 0], data[:, 1], color="red", label="Dados reais", zorder=5)
plt.xlabel("Tensão (V)")
plt.ylabel("Corrente (A)")
plt.title("Curva I-V do módulo fotovoltaico (1 diodo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()