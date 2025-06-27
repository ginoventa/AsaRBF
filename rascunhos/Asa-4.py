import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. CARREGAR E NORMALIZAR DADOS ===
df = pd.read_excel("dados/dados-asa.xls", engine='xlrd')
x1 = df.iloc[:, 0].values
x2 = df.iloc[:, 1].values
z  = df.iloc[:, 2].values

X_raw = np.column_stack((x1, x2))
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)
n_pontos = X.shape[0]

# === 2. CENTROS DAS RBFs ===
np.random.seed(0)
n_centros = 150
indices = np.random.choice(n_pontos, n_centros, replace=False)
centros = X[indices]

# === 3. FUNÇÃO DE BASE GAUSSIANA ===
def phi(x, c, alpha=0.05):
    return np.exp(-alpha * np.sum((x - c)**2, axis=1))

# === 4. MATRIZ Φ ===
alpha = 0.05
Phi = np.zeros((n_pontos, n_centros))
for j, c in enumerate(centros):
    Phi[:, j] = phi(X, c, alpha)

# === 5. MÍNIMOS QUADRADOS ===
a, *_ = np.linalg.lstsq(Phi, z, rcond=None)

# === 6. MALHA (GRID) FINA E NORMALIZADA ===
x1_lin = np.linspace(min(x1), max(x1), 100)
x2_lin = np.linspace(min(x2), max(x2), 100)
X1g, X2g = np.meshgrid(x1_lin, x2_lin)
grid_raw = np.column_stack((X1g.ravel(), X2g.ravel()))
grid = (grid_raw - X_min) / (X_max - X_min)

# === 7. AVALIAÇÃO DO MODELO NA MALHA ===
Z_pred = np.zeros(len(grid))
for j, c in enumerate(centros):
    Z_pred += a[j] * phi(grid, c, alpha)
Z_grid = Z_pred.reshape(X1g.shape)

# === 8. FUNÇÃO DE PLOTAGEM COM PONTOS ORIGINAIS ===
def plot_asa(angle, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Superfície
    ax.plot_surface(X1g, X2g, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Pontos reais
    ax.scatter(x1, x2, z, color='blue', s=5, label="Pontos reais")

    ax.set_xlabel('x₁ (longitudinal)')
    ax.set_ylabel('x₂ (lateral)')
    ax.set_zlabel('z (altura)')
    ax.set_title(title)
    ax.view_init(*angle)
    
    # Preservar proporção visual da asa
    ax.set_box_aspect([np.ptp(x1), np.ptp(x2), np.ptp(z)])
    ax.legend()

# === 9. TRÊS VISTAS DA ASA ===
plot_asa((30, 45), "Vista 1 - Padrão")
plot_asa((20, 120), "Vista 2 - Lateral")
plot_asa((90, 0), "Vista 3 - Topo")

plt.show()
