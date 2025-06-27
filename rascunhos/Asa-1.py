import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === 1. CARREGAR OS DADOS ===
excel_path = None
if os.path.exists("dados/dados-asa.xlsx"):
    excel_path = "dados/dados-asa.xlsx"
elif os.path.exists("dados/dados-asa.xls"):
    excel_path = "dados/dados-asa.xls"
else:
    raise FileNotFoundError("Arquivo de dados não encontrado em 'dados/dados-asa.xlsx' ou 'dados/dados-asa.xls'")
df = pd.read_excel(excel_path)
x1 = df.iloc[:, 0].values  # eixo longitudinal
x2 = df.iloc[:, 1].values  # eixo lateral
z  = df.iloc[:, 2].values  # altura

# Normalização dos dados de entrada
x1_mean, x1_std = np.mean(x1), np.std(x1)
x2_mean, x2_std = np.mean(x2), np.std(x2)
x1n = (x1 - x1_mean) / x1_std
x2n = (x2 - x2_mean) / x2_std
X = np.column_stack((x1n, x2n))  # pontos (x, y) normalizados

# === 2. SELEÇÃO DOS CENTROS RBF ===
n_centros = min(50, len(X))  # reduzido para evitar instabilidade
indices = np.random.choice(len(X), n_centros, replace=False)
centros = X[indices, :]

# === 3. DEFINIÇÃO DA RBF GAUSSIANA ===
def rbf_gaussiana(x, c, alpha):
    dist2 = np.sum((x - c) ** 2, axis=1)
    return np.exp(-alpha * dist2)

# === 4. CONSTRUÇÃO DA MATRIZ Φ ===
# alpha adaptativo
if n_centros > 1:
    dists = np.linalg.norm(centros[None, :, :] - centros[:, None, :], axis=2)
    med_dist = np.median(dists[dists > 0])
    alpha = 1.0 / (2 * med_dist**2)
else:
    alpha = 1.0
Phi = np.zeros((X.shape[0], n_centros))
for j in range(n_centros):
    Phi[:, j] = rbf_gaussiana(X, centros[j], alpha)

# === 5. RESOLVER POR MÍNIMOS QUADRADOS ===
a, _, _, _ = np.linalg.lstsq(Phi, z, rcond=None)

# === 6. GERAÇÃO DA MALHA PARA AVALIAÇÃO DO MODELO ===
x1_lin = np.linspace(np.min(x1), np.max(x1), 100)
x2_lin = np.linspace(np.min(x2), np.max(x2), 100)
X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
# Normalizar a malha
X1n_grid = (X1_grid - x1_mean) / x1_std
X2n_grid = (X2_grid - x2_mean) / x2_std
grid_points = np.column_stack((X1n_grid.ravel(), X2n_grid.ravel()))

# Avaliar P(x) = ∑ a_j φ_j(x)
P_vals = np.zeros(grid_points.shape[0])
for j in range(n_centros):
    P_vals += a[j] * rbf_gaussiana(grid_points, centros[j], alpha)

Z_grid = P_vals.reshape(X1_grid.shape)

# === 7. PLOTAGEM DAS TRÊS VISTAS ===
def plot_surface(X1, X2, Z, angle, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x (longitudinal)')
    ax.set_ylabel('y (lateral)')
    ax.set_zlabel('z (altura)')
    ax.view_init(*angle)
    ax.set_title(title)

plot_surface(X1_grid, X2_grid, Z_grid, (30, 45), "Vista 1 - Padrão")
plot_surface(X1_grid, X2_grid, Z_grid, (20, 60), "Vista 2 - Inclinada")
plot_surface(X1_grid, X2_grid, Z_grid, (90, 0), "Vista 3 - Topo")

plt.show()
