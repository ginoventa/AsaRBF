import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. Carregar dados
# Usa MinMaxScaler para normalização padrão

# Leitura dos dados
# (ajusta para aceitar tanto .xls quanto .xlsx)
import os
if os.path.exists("dados/dados-asa.xls"):
    df = pd.read_excel("dados/dados-asa.xls", engine="xlrd")
elif os.path.exists("dados/dados-asa.xlsx"):
    df = pd.read_excel("dados/dados-asa.xlsx")
else:
    raise FileNotFoundError("Arquivo de dados não encontrado.")
x1 = df.iloc[:, 0].values  # eixo x
x2 = df.iloc[:, 1].values  # eixo y
z  = df.iloc[:, 2].values  # eixo z (altura)

# 2. Normalização MinMax
scaler = MinMaxScaler()
X = np.column_stack((x1, x2))
X_scaled = scaler.fit_transform(X)
z_scaled = (z - np.mean(z)) / np.std(z)

# 3. Seleção dos centros via KMeans
k = 150  # número de centros
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
centros_scaled = kmeans.cluster_centers_

# 4. Função RBF Gaussiana
# (mantém broadcasting correto)
def rbf_gauss(x, c, alpha):
    return np.exp(-alpha * np.sum((x - c)**2, axis=1))

# 5. Cálculo de alpha dinâmico
from scipy.spatial.distance import pdist
mean_dist = np.mean(pdist(centros_scaled))
alpha = 1 / (2 * mean_dist**2)

# 6. Construção da matriz Phi
Phi = np.zeros((X_scaled.shape[0], centros_scaled.shape[0]))
for j in range(centros_scaled.shape[0]):
    Phi[:, j] = rbf_gauss(X_scaled, centros_scaled[j], alpha)

# 7. Resolução dos coeficientes com regularização ridge
lamb = 1e-6
A = Phi.T @ Phi + lamb * np.eye(Phi.shape[1])
b = Phi.T @ z_scaled
coef = np.linalg.solve(A, b)

# 8. Geração de grid para visualização
x1_vals = np.linspace(np.min(x1), np.max(x1), 100)
x2_vals = np.linspace(np.min(x2), np.max(x2), 100)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)
grid_pts = np.column_stack((X1g.ravel(), X2g.ravel()))
grid_scaled = scaler.transform(grid_pts)

# 9. Avaliação do modelo na malha
def P(x, centros, a, alpha):
    phi = np.exp(-alpha * np.linalg.norm(x - centros, axis=1)**2)
    return np.dot(a, phi)

Zg = np.array([P(pt, centros_scaled, coef, alpha) for pt in grid_scaled])
Zg = Zg.reshape(X1g.shape)
# Desnormaliza z para visualização
Zg = Zg * np.std(z) + np.mean(z)

# 10. Visualização: superfície + pontos reais
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1g, X2g, Zg, cmap='viridis', alpha=0.8, antialiased=True)
ax.scatter(x1, x2, z, color='blue', s=10, label='Dados reais')
ax.set_xlabel('x₁ (longitudinal)')
ax.set_ylabel('x₂ (lateral)')
ax.set_zlabel('z (altura)')
ax.set_title('Superfície modelada + pontos reais')
ax.legend()

# Ajuste de proporção dos eixos
x_range = np.ptp(x1)
y_range = np.ptp(x2)
z_range = np.ptp(z)
max_range = max(x_range, y_range, z_range) / 2.0
mid_x = np.mean(x1)
mid_y = np.mean(x2)
mid_z = np.mean(z)
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()