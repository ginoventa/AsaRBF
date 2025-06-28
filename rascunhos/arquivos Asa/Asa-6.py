import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
import os

# === 1. Leitura dos dados ===
df = pd.read_excel('dados/dados-asa.xls')  # ou .xlsx se for o caso
x1 = df['x'].values  # eixo longitudinal
x2 = df['y'].values  # eixo lateral
z = df['z'].values   # altura

X = np.vstack((x1, x2)).T

# === 2. Normalização ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Uso de todos os pontos como centros ===
centros_scaled = X_scaled

# === 4. Cálculo de alpha ===
dists = pdist(centros_scaled)
mean_dist = np.mean(dists)
alpha = 1 / (2 * mean_dist**2)

# === 5. RBF Gaussiana ===
def rbf_gaussiana(x, c, alpha):
    return np.exp(-alpha * np.linalg.norm(x - c, axis=1)**2)

# === 6. Construção de Phi ===
Phi = np.zeros((len(X_scaled), len(centros_scaled)))
for i in range(len(X_scaled)):
    Phi[i] = rbf_gaussiana(X_scaled[i], centros_scaled, alpha)

# === 7. Resolver o sistema linear ===
a = np.linalg.solve(Phi, z)

# === 8. Modelo P(x) ===
def P(x, centros, a, alpha):
    phi = np.exp(-alpha * np.linalg.norm(x - centros, axis=1)**2)
    return np.dot(a, phi)

# === 9. Geração da malha (grid) ===
x1_grid = np.linspace(min(x1), max(x1), 150)
x2_grid = np.linspace(min(x2), max(x2), 150)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)
grid_points = np.vstack((X1g.ravel(), X2g.ravel())).T
grid_scaled = scaler.transform(grid_points)

# === 10. Avaliação do modelo nos pontos do grid ===
Zg = np.array([P(pt, centros_scaled, a, alpha) for pt in grid_scaled])
Zg = Zg.reshape(X1g.shape)

# === 11. Visualizações ===

# Vista 1 - Padrão
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X1g, X2g, Zg, cmap='viridis', rstride=1, cstride=1, antialiased=True)
ax1.scatter(x1, x2, z, color='r', s=5, label='Dados reais')
ax1.set_title('Vista 1 - Padrão')
ax1.set_xlabel('x₁ (longitudinal)')
ax1.set_ylabel('x₂ (lateral)')
ax1.set_zlabel('z (altura)')
ax1.legend()
plt.tight_layout()
plt.show()

# Vista 2 - Lateral
fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X1g, X2g, Zg, cmap='plasma', rstride=1, cstride=1, antialiased=True)
ax2.view_init(azim=90, elev=20)
ax2.set_title('Vista 2 - Lateral')
ax2.set_xlabel('x₁ (longitudinal)')
ax2.set_ylabel('x₂ (lateral)')
ax2.set_zlabel('z (altura)')
plt.tight_layout()
plt.show()

# Vista 3 - Topo
fig3 = plt.figure(figsize=(7, 5))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X1g, X2g, Zg, cmap='coolwarm', rstride=1, cstride=1, antialiased=True)
ax3.view_init(azim=90, elev=90)
ax3.set_title('Vista 3 - Topo')
ax3.set_xlabel('x₁ (longitudinal)')
ax3.set_ylabel('x₂ (lateral)')
ax3.set_zlabel('z (altura)')
plt.tight_layout()
plt.show()