import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. Leitura dos dados ===
df = pd.read_excel('dados-asa.xls')

x1 = df['x'].values  # eixo longitudinal
x2 = df['y'].values  # eixo lateral
z = df['z'].values   # altura (z = y no modelo)

X = np.vstack((x1, x2)).T  # pontos x_k no plano

# === 2. Seleção dos centros da RBF ===
np.random.seed(42)
idx_centros = np.random.choice(len(X), 200, replace=False)
centros = X[idx_centros]

# === 3. Definição da função RBF Gaussiana ===
def rbf_gaussiana(x, c, alpha=1.0):
    return np.exp(-alpha * np.linalg.norm(x - c, axis=1)**2)

# === 4. Construção da matriz Phi ===
alpha = 1.0
Phi = np.zeros((len(X), len(centros)))
for i in range(len(X)):
    Phi[i] = rbf_gaussiana(X[i], centros, alpha)

# === 5. Ajuste dos coeficientes a por mínimos quadrados ===
a = np.linalg.lstsq(Phi, z, rcond=None)[0]

# === 6. Definição do modelo P(x) ===
def P(x, centros, a, alpha=1.0):
    phi = np.exp(-alpha * np.linalg.norm(x - centros, axis=1)**2)
    return np.dot(a, phi)

# === 7. Geração da malha fina (grid) ===
x1_grid = np.linspace(min(x1), max(x1), 100)
x2_grid = np.linspace(min(x2), max(x2), 100)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)

Zg = np.zeros_like(X1g)
for i in range(X1g.shape[0]):
    for j in range(X1g.shape[1]):
        x_point = np.array([X1g[i, j], X2g[i, j]])
        Zg[i, j] = P(x_point, centros, a, alpha)

# === 8. Visualizações ===
fig = plt.figure(figsize=(15, 5))

# Vista 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1g, X2g, Zg, cmap='viridis')
ax1.set_title('Vista Frontal')

# Vista 2
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X1g, X2g, Zg, cmap='plasma')
ax2.view_init(azim=45, elev=30)
ax2.set_title('Vista em Perspectiva')

# Vista 3
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X1g, X2g, Zg, cmap='coolwarm')
ax3.view_init(azim=90, elev=20)
ax3.set_title('Vista Lateral')

plt.tight_layout()
plt.show()