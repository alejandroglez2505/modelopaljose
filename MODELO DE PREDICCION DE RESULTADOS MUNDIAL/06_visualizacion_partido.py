import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CARGAR RESULTADOS
# =====================================================

print("Cargando probabilidades...")

df = pd.read_csv(
    "probabilidades_partido.csv"
)

# =====================================================
# CONSTRUIR MATRIZ
# =====================================================

max_goles = 6

matriz = np.zeros(
    (
        max_goles + 1,
        max_goles + 1
    )
)

for _, row in df.iterrows():

    marcador = row["marcador"]

    goles_local = int(
        marcador.split("-")[0]
    )

    goles_visitante = int(
        marcador.split("-")[1]
    )

    if (
        goles_local <= max_goles
        and
        goles_visitante <= max_goles
    ):

        matriz[
            goles_local,
            goles_visitante
        ] = row["probabilidad"]

# =====================================================
# HEATMAP
# =====================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    matriz,
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label="Probabilidad"
)

plt.title(
    "Heatmap de Marcadores"
)

plt.xlabel(
    "Goles Visitante"
)

plt.ylabel(
    "Goles Local"
)

plt.xticks(
    range(max_goles + 1)
)

plt.yticks(
    range(max_goles + 1)
)

plt.tight_layout()

plt.savefig(
    "heatmap_resultados.png",
    dpi=300
)

print(
    "Generado: heatmap_resultados.png"
)

# =====================================================
# TOP 10 MARCADORES
# =====================================================

top10 = df.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(

    top10["marcador"][::-1],

    top10["probabilidad"][::-1]

)

plt.title(
    "Top 10 Marcadores"
)

plt.xlabel(
    "Probabilidad"
)

plt.tight_layout()

plt.savefig(
    "top10_marcadores.png",
    dpi=300
)

print(
    "Generado: top10_marcadores.png"
)

# =====================================================
# MOSTRAR
# =====================================================

plt.show()