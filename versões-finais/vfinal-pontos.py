import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar dados
df = pd.read_excel("dados/dados-asa.xls", engine="xlrd")
x1 = df.iloc[:, 0].values  # eixo x
x2 = df.iloc[:, 1].values  # eixo y
z  = df.iloc[:, 2].values  # eixo z (altura)

# 2. Normalização
def normaliza(v):
    return (v - np.mean(v)) / np.std(v), np.mean(v), np.std(v)
x1n, x1m, x1s = normaliza(x1)
x2n, x2m, x2s = normaliza(x2)
zn, zm, zs = normaliza(z)
X = np.column_stack((x1n, x2n))

# 3. Seleção dos centros (exemplo: todos os pontos)
centros = X.copy()
n_centros = X.shape[0]

# 4. Função RBF Gaussiana
def rbf_gauss(x, c, alpha):
    # x: (N, 2), c: (2,), retorna (N,)
    # Corrigido para garantir broadcasting correto
    return np.exp(-alpha * np.sum((x - c)**2, axis=1))

alpha = 0.05  # ajuste conforme necessário

# 5. Construção da matriz Phi
Phi = np.zeros((X.shape[0], n_centros))
for j in range(n_centros):
    Phi[:, j] = rbf_gauss(X, centros[j], alpha)

# 6. Resolução dos coeficientes
a, *_ = np.linalg.lstsq(Phi, zn, rcond=None)

# 7. Geração de grid fino para visualização
x1_vals = np.linspace(np.min(x1), np.max(x1), 100)
x2_vals = np.linspace(np.min(x2), np.max(x2), 100)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)
X1g_n = (X1g - x1m) / x1s
X2g_n = (X2g - x2m) / x2s
grid_pts = np.column_stack((X1g_n.ravel(), X2g_n.ravel()))

# 8. Avaliação do modelo na malha
Z = np.zeros(grid_pts.shape[0])
for j in range(n_centros):
    Z += a[j] * rbf_gauss(grid_pts, centros[j], alpha)
Zg = Z.reshape(X1g.shape) * zs + zm  # desnormaliza z

# --- NOVO: calcular o contorno modelado da asa ---
# Para cada ponto do contorno (x1, x2), calcular z_modelado usando a RBF
X_contorno_n = np.column_stack(((x1 - x1m) / x1s, (x2 - x2m) / x2s))
z_contorno_modelado = np.zeros_like(x1)
for j in range(n_centros):
    z_contorno_modelado += a[j] * rbf_gauss(X_contorno_n, centros[j], alpha)
z_contorno_modelado = z_contorno_modelado * zs + zm  # desnormaliza

# Visualização única, grande, com superfície e pontos reais
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(x1, x2, z, color='blue', s=20)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('z')
ax.set_title('Vista 1')
ax.view_init(elev=22, azim=130)
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