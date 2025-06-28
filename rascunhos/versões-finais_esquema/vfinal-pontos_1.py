import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
import os

def main():
    # ETAPA 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
    # -------------------------------------------------
    # O objetivo desta etapa é ler os pontos da asa (x, y, z) de um arquivo Excel
    # e prepará-los para o modelo.

    try:
        # Tenta localizar o arquivo de dados, seja no formato .xls ou .xlsx.
        # O 'engine="xlrd"' é necessário para o formato antigo .xls.
        if os.path.exists("dados/dados-asa.xls"):
            df = pd.read_excel("dados/dados-asa.xls", engine="xlrd")
        elif os.path.exists("dados/dados-asa.xlsx"):
            df = pd.read_excel("dados/dados-asa.xlsx")
        else:
            raise FileNotFoundError("Arquivo de dados 'dados-asa.xls' ou 'dados-asa.xlsx' não encontrado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo de dados: {e}")
        return

    # Extrai as colunas do DataFrame do pandas para arrays NumPy.
    x1 = df.iloc[:, 0].values  # Coordenada longitudinal
    x2 = df.iloc[:, 1].values  # Coordenada lateral
    z = df.iloc[:, 2].values   # Altura

    # ETAPA 2: NORMALIZAÇÃO DOS DADOS
    # ----------------------------------
    # A normalização coloca todas as variáveis na mesma escala, o que é crucial
    # para algoritmos baseados em distância, como RBF e KMeans.

    # Para as coordenadas (x1, x2), usamos MinMaxScaler. Ele transforma os dados
    # para que fiquem no intervalo [0, 1]. Isso preserva a forma da distribuição.
    scaler = MinMaxScaler()
    X = np.column_stack((x1, x2))
    X_scaled = scaler.fit_transform(X)

    # Para a altura (z), usamos a normalização Z-score (subtrair a média e dividir
    # pelo desvio padrão). Isso centraliza os dados em torno de 0.
    z_mean, z_std = np.mean(z), np.std(z)
    z_scaled = (z - z_mean) / z_std

    # ETAPA 3: SELEÇÃO DE CENTROS RBF COM KMEANS
    # ---------------------------------------------
    # O modelo RBF precisa de "centros" para as funções de base radial. Em vez de usar
    # todos os pontos de dados (o que seria computacionalmente caro), usamos o KMeans
    # para encontrar um número menor de centros representativos (k=150).
    # 'n_init=10' executa o algoritmo 10 vezes com diferentes sementes para encontrar um resultado estável.
    k = 150
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    centros_scaled = kmeans.cluster_centers_

    # ETAPA 4: CÁLCULO DO PARÂMETRO ALPHA (LARGURA) DA RBF
    # ------------------------------------------------------
    # A função RBF Gaussiana precisa de um parâmetro 'alpha' que controla a "largura" ou
    # influência de cada centro. Uma heurística comum é calculá-lo com base na
    # distância média entre os centros para garantir uma boa sobreposição.
    # 'pdist' calcula a distância pareada entre todos os centros.
    mean_dist = np.mean(pdist(centros_scaled))
    alpha = 1 / (2 * mean_dist**2)

    # Define a função RBF Gaussiana.
    def rbf_gauss(x, c, alpha_val):
        # Calcula a distância Euclidiana ao quadrado entre os pontos (x) e os centros (c)
        # e aplica a função exponencial.
        return np.exp(-alpha_val * np.sum((x[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))

    # ETAPA 5: CONSTRUÇÃO DO MODELO RBF
    # -----------------------------------
    # Aqui, construímos o sistema de equações lineares (A * w = b) que precisamos resolver
    # para encontrar os pesos (coeficientes) do modelo RBF.

    # A matriz Phi (ou matriz de design) é calculada aplicando a função RBF
    # a cada ponto de dado em relação a cada centro.
    Phi = rbf_gauss(X_scaled, centros_scaled, alpha)

    # Para evitar sobreajuste (overfitting), adicionamos um termo de regularização de Ridge (L2).
    # O parâmetro 'lamb' controla a força da regularização. Um valor maior cria uma superfície mais suave.
    lamb = 0.1
    A = Phi.T @ Phi + lamb * np.eye(Phi.shape[1])
    b = Phi.T @ z_scaled

    # Resolvemos o sistema linear para encontrar os coeficientes (pesos) do modelo.
    coef = np.linalg.solve(A, b)

    # ETAPA 6: GERAÇÃO DE UMA MALHA (GRID) PARA VISUALIZAÇÃO
    # ---------------------------------------------------------
    # Para visualizar a superfície 3D contínua, criamos uma malha regular de pontos (x, y)
    # e usamos nosso modelo treinado para prever a altura (z) em cada um desses pontos.

    # 'np.linspace' cria vetores de pontos igualmente espaçados.
    x1_vals = np.linspace(np.min(x1), np.max(x1), 100)
    x2_vals = np.linspace(np.min(x2), np.max(x2), 100)
    # 'np.meshgrid' cria matrizes de coordenadas a partir dos vetores.
    X1g, X2g = np.meshgrid(x1_vals, x2_vals)
    grid_pts = np.column_stack((X1g.ravel(), X2g.ravel()))
    # Normalizamos a malha usando o mesmo 'scaler' dos dados de treino.
    grid_scaled = scaler.transform(grid_pts)

    # ETAPA 7: AVALIAÇÃO DO MODELO NA MALHA
    # ---------------------------------------
    # Usamos o modelo treinado (os coeficientes) para calcular a altura na malha.
    Z_grid_scaled = rbf_gauss(grid_scaled, centros_scaled, alpha) @ coef
    # Desnormalizamos o resultado para obter a altura na escala original.
    Zg = Z_grid_scaled.reshape(X1g.shape) * z_std + z_mean

    # ETAPA 8: VISUALIZAÇÃO DOS RESULTADOS
    # --------------------------------------
    # Usamos o Matplotlib para criar os gráficos 3D.

    # 'plt.figure' cria a janela da figura.
    fig = plt.figure(figsize=(24, 8))

    # --- Vista 1: Padrão ---
    # 'fig.add_subplot' adiciona um eixo (um gráfico individual) à figura.
    ax1 = fig.add_subplot(131, projection='3d')
    # 'plot_surface' desenha a superfície 3D. 'cmap' define o mapa de cores.
    ax1.plot_surface(X1g, X2g, Zg, cmap='viridis', alpha=0.8, antialiased=True, rstride=1, cstride=1)
    # 'scatter' plota os pontos de dados originais em vermelho.
    ax1.scatter(x1, x2, z, color='red', s=10, label='Dados Reais', depthshade=False)
    ax1.set_title('Vista 1 - Padrão', fontsize=14)
    # 'view_init' ajusta o ângulo de visão da câmera (elevação e azimute).
    ax1.view_init(elev=30, azim=135)

    # --- Vista 2: Lateral ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1g, X2g, Zg, cmap='plasma', alpha=0.8, antialiased=True, rstride=1, cstride=1)
    ax2.scatter(x1, x2, z, color='red', s=10, label='Dados Reais', depthshade=False)
    ax2.set_title('Vista 2 - Lateral', fontsize=14)
    ax2.view_init(elev=20, azim=90)

    # --- Vista 3: Topo ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X1g, X2g, Zg, cmap='coolwarm', alpha=0.8, antialiased=True, rstride=1, cstride=1)
    ax3.scatter(x1, x2, z, color='red', s=10, label='Dados Reais', depthshade=False)
    ax3.set_title('Vista 3 - Topo', fontsize=14)
    ax3.view_init(elev=90, azim=-90)

    # Loop para configurar rótulos e legendas para todos os subplots.
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Eixo Longitudinal (x₁)', fontsize=10)
        ax.set_ylabel('Eixo Lateral (x₂)', fontsize=10)
        ax.set_zlabel('Altura (z)', fontsize=10)
        ax.legend()
        # 'autoscale_view' permite que o Matplotlib ajuste os limites dos eixos
        # para uma melhor proporção visual, em vez de forçar uma escala igual.
        ax.autoscale_view()

    # 'plt.tight_layout' ajusta o espaçamento entre os gráficos.
    plt.tight_layout()
    # 'plt.show' exibe a janela do gráfico.
    plt.show()

# Bloco padrão do Python: o código dentro deste 'if' só executa
# quando o script é rodado diretamente (e não quando é importado por outro script).
if __name__ == "__main__":
    main()