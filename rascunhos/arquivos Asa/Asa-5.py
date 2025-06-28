import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carregando dados (308 pontos)
df = pd.read_excel("dados/dados-asa.xls", engine="xlrd")
x1 = df.iloc[:, 0].values  # eixo longitudinal
x2 = df.iloc[:, 1].values  # eixo lateral
z  = df.iloc[:, 2].values  # altura

# Normalização dos dados de entrada
def normaliza(v):
    return (v - np.mean(v)) / np.std(v), np.mean(v), np.std(v)
x1n, x1m, x1s = normaliza(x1)
x2n, x2m, x2s = normaliza(x2)
z_n, zm, zs = normaliza(z)
X = np.column_stack((x1n, x2n))

# 1b. Conferência dos dados
print('Primeiros valores de x1:', x1[:10])
print('Primeiros valores de x2:', x2[:10])
print('Primeiros valores de z:', z[:10])

# Gráfico com projeção dos pontos no plano x1-x2 (z=0)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, z, color='blue', s=10, label='Pontos reais')
ax.scatter(x1, x2, np.zeros_like(z), color='red', s=10, alpha=0.5, label='Projeção no plano x1-x2')
ax.set_xlabel('x₁ (longitudinal)')
ax.set_ylabel('x₂ (lateral)')
ax.set_zlabel('z (altura)')
ax.set_title('Pontos reais e sua projeção no plano x₁-x₂')
ax.view_init(elev=22, azim=130)
ax.legend()
plt.tight_layout()
plt.show()
