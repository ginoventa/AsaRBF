import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# === 1. Leitura dos dados ===
df = pd.read_excel('dados-asa.xls')

x1 = df['x'].values  # eixo longitudinal
x2 = df['y'].values  # eixo lateral
z = df['z'].values   # altura (z = y no modelo)

X = np.vstack((x1, x2)).T  # pontos x_k no plano

# === 2. Normalização dos dados ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Seleção dos centros usando KMeans ===
kmeans = KMeans(n_clusters=200, random_state=42).fit(X_scaled)
centros_scaled = kmeans.cluster_centers_

# === 4. Definição da função RBF Gaussiana ===
def rbf_gaussiana(x, c, alpha=50.0):
    return np.exp(-alpha * np.linalg.norm(x - c, axis=1)**2)

# === 5. Construção da matriz Phi ===
alpha = 50.0
Phi = np.zeros((len(X_scaled), len(centros_scaled)))
for i in range(len(X_scaled)):
    Phi[i] = rbf_gaussiana(X_scaled[i], centros_scaled, alpha)

# === 6. Ajuste dos coeficientes a por mínimos quadrados ===
a = np.linalg.lstsq(Phi, z, rcond=None)[0]

# === 7. Definição do modelo P(x) ===
def P(x, centros, a, alpha=50.0):
    phi = np.exp(-alpha * np.linalg.norm(x - centros, axis=1)**2)
    return np.dot(a, phi)

# === 8. Geração da malha fina (grid) ===
x1_grid = np.linspace(min(x1), max(x1), 100)
x2_grid = np.linspace(min(x2), max(x2), 100)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)

# Transformar grid para o mesmo espaço escalado
grid_points = np.vstack((X1g.ravel(), X2g.ravel())).T
grid_scaled = scaler.transform(grid_points)

Zg = np.array([P(pt, centros_scaled, a, alpha) for pt in grid_scaled])
Zg = Zg.reshape(X1g.shape)

# === 9. Visualizações ===
fig = plt.figure(figsize=(15, 5))

# Vista 1 - Padrão
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1g, X2g, Zg, cmap='viridis')
ax1.set_title('Vista 1 - Padrão')
ax1.set_xlabel('x (longitudinal)')
ax1.set_ylabel('y (lateral)')
ax1.set_zlabel('z (altura)')

# Vista 2 - Inclinada
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X1g, X2g, Zg, cmap='plasma')
ax2.view_init(azim=45, elev=30)
ax2.set_title('Vista 2 - Inclinada')
ax2.set_xlabel('x (longitudinal)')
ax2.set_ylabel('y (lateral)')
ax2.set_zlabel('z (altura)')

# Vista 3 - Topo
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X1g, X2g, Zg, cmap='coolwarm')
ax3.view_init(azim=90, elev=90)
ax3.set_title('Vista 3 - Topo')
ax3.set_xlabel('x (longitudinal)')
ax3.set_ylabel('y (lateral)')
ax3.set_zlabel('z (altura)')

plt.tight_layout()
plt.show()