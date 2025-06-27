import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carregando dados (308 pontos)
df = pd.read_excel("dados/dados-asa.xls", engine="xlrd")
x1 = df.iloc[:, 0].values  # eixo longitudinal
x2 = df.iloc[:, 1].values  # eixo lateral
z  = df.iloc[:, 2].values  # altura
X = np.column_stack((x1, x2))

# 2. Definindo os centros: todos os 308 pontos
centros = X.copy()
N = centros.shape[0]

# 3. Função Gaussiana
def rbf(x, c, alpha):
    # x é [M x 2], c é centro [2]
    return np.exp(-alpha * np.sum((x - c)**2, axis=1))

# 4. Construindo matriz Phi
alpha = 1e-5  # ajustado para não gerar deformações
Phi = np.zeros((X.shape[0], N))
for j in range(N):
    Phi[:, j] = rbf(X, centros[j], alpha)

# 5. Resolvendo mínimos quadrados
a, *_ = np.linalg.lstsq(Phi, z, rcond=None)

# 6. Criando malha fina para visualizar P(x)
x1_vals = np.linspace(np.min(x1), np.max(x1), 100)
x2_vals = np.linspace(np.min(x2), np.max(x2), 100)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)
grid_pts = np.column_stack((X1g.ravel(), X2g.ravel()))

# 7. Avaliando o modelo na malha
Z = np.zeros(grid_pts.shape[0])
for j in range(N):
    Z += a[j] * rbf(grid_pts, centros[j], alpha)
Zg = Z.reshape(X1g.shape)

# 8. Visualização com pontos reais
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1g, X2g, Zg, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
ax.scatter(x1, x2, z, color='blue', s=5)

ax.set_xlabel("x₁ (longitudinal)")
ax.set_ylabel("x₂ (lateral)")
ax.set_zlabel("z (altura)")
ax.set_title("Superfície da asa modelada com RBF - Vista 1")
ax.view_init(elev=22, azim=130)  # igual ao seu ângulo

plt.tight_layout()
plt.show()
