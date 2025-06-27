import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. CARREGAR E NORMALIZAR OS DADOS ===
df = pd.read_excel("dados/dados-asa.xls", engine='xlrd')  # Para .xls
x1 = df.iloc[:, 0].values
x2 = df.iloc[:, 1].values
z  = df.iloc[:, 2].values

X_raw = np.column_stack((x1, x2))

# Normalização min-max
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)

n_pontos = X.shape[0]

# === 2. SELEÇÃO DOS CENTROS RBF ===
np.random.seed(0)
n_centros = 150
indices = np.random.choice(n_pontos, n_centros, replace=False)
centros = X[indices]

# === 3. FUNÇÃO DE BASE RADIAL GAUSSIANA ===
def phi(x, c, alpha=0.05):
    return np.exp(-alpha * np.sum((x - c)**2, axis=1))

# === 4. CONSTRUÇÃO DA MATRIZ Φ ===
alpha = 0.05
Phi = np.zeros((n_pontos, n_centros))
for j, c in enumerate(centros):
    Phi[:, j] = phi(X, c, alpha)

# === 5. RESOLUÇÃO POR MÍNIMOS QUADRADOS ===
a, *_ = np.linalg.lstsq(Phi, z, rcond=None)

# === 6. GERAÇÃO DA MALHA NORMALIZADA ===
x1_lin = np.linspace(min(x1), max(x1), 80)
x2_lin = np.linspace(min(x2), max(x2), 80)
X1g, X2g = np.meshgrid(x1_lin, x2_lin)
grid_points_raw = np.column_stack((X1g.ravel(), X2g.ravel()))

# Normaliza também a malha
grid_points = (grid_points_raw - X_min) / (X_max - X_min)

# === 7. AVALIAÇÃO DO MODELO NA MALHA ===
Z_pred = np.zeros(len(grid_points))
for j, c in enumerate(centros):
    Z_pred += a[j] * phi(grid_points, c, alpha)
Z_grid = Z_pred.reshape(X1g.shape)

# === 8. VISUALIZAÇÕES ===
def plot_asa(angle, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1g, X2g, Z_grid, cmap='plasma', edgecolor='none')
    ax.set_xlabel('x₁ (longitudinal)')
    ax.set_ylabel('x₂ (lateral)')
    ax.set_zlabel('z (altura)')
    ax.view_init(*angle)
    ax.set_title(title)

plot_asa((30, 45), "Vista 1 - Padrão")
plot_asa((20, 120), "Vista 2 - Inclinada")
plot_asa((90, 0), "Vista 3 - Topo")

plt.show()
