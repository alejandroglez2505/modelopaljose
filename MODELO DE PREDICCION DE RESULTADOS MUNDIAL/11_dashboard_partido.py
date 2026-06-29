import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# =====================================================
# CONFIG
# =====================================================

N_SIMULACIONES = 100000
MAX_GOLES = 6

# =====================================================
# CARGAR ARCHIVOS
# =====================================================

print("Cargando datos...")

features = pd.read_csv(
    "features_equipos.csv"
)

ratings_pymc = pd.read_csv(
    "ratings_pymc.csv"
)

with open(
    "modelo_home_xgb.pkl",
    "rb"
) as f:

    modelo_home = pickle.load(f)

with open(
    "modelo_away_xgb.pkl",
    "rb"
) as f:

    modelo_away = pickle.load(f)

# =====================================================
# EQUIPOS
# =====================================================

equipos = sorted(
    features["equipo"].unique()
)

print("\n========================")
print("EQUIPOS DISPONIBLES")
print("========================\n")

for i, equipo in enumerate(
    equipos,
    start=1
):
    print(
        f"{i:2d}. {equipo}"
    )

idx_a = int(
    input(
        "\nNumero Equipo A: "
    )
)

idx_b = int(
    input(
        "Numero Equipo B: "
    )
)

EQUIPO_A = equipos[
    idx_a - 1
]

EQUIPO_B = equipos[
    idx_b - 1
]

# =====================================================
# FUNCION CREAR FILA
# =====================================================

def crear_fila(
    local,
    visitante
):

    home = features[
        features["equipo"] == local
    ]

    away = features[
        features["equipo"] == visitante
    ]

    home_pymc = ratings_pymc[
        ratings_pymc["equipo"] == local
    ]

    away_pymc = ratings_pymc[
        ratings_pymc["equipo"] == visitante
    ]

    fila = pd.DataFrame({

        "neutral":[1],
        "dias":[0],
        "peso_tiempo":[1],
        "peso_torneo":[1],
        "peso_total":[1],

        "home_ataque":[home.iloc[0]["ataque"]],
        "home_defensa":[home.iloc[0]["defensa"]],
        "home_diferencia_goles":[home.iloc[0]["diferencia_goles"]],
        "home_forma_5":[home.iloc[0]["forma_5"]],
        "home_forma_10":[home.iloc[0]["forma_10"]],
        "home_forma_30":[home.iloc[0]["forma_30"]],
        "home_puntos_ponderados":[home.iloc[0]["puntos_ponderados"]],
        "home_victorias":[home.iloc[0]["victorias"]],
        "home_empates":[home.iloc[0]["empates"]],
        "home_derrotas":[home.iloc[0]["derrotas"]],
        "home_elo":[home.iloc[0]["elo"]],

        "away_ataque":[away.iloc[0]["ataque"]],
        "away_defensa":[away.iloc[0]["defensa"]],
        "away_diferencia_goles":[away.iloc[0]["diferencia_goles"]],
        "away_forma_5":[away.iloc[0]["forma_5"]],
        "away_forma_10":[away.iloc[0]["forma_10"]],
        "away_forma_30":[away.iloc[0]["forma_30"]],
        "away_puntos_ponderados":[away.iloc[0]["puntos_ponderados"]],
        "away_victorias":[away.iloc[0]["victorias"]],
        "away_empates":[away.iloc[0]["empates"]],
        "away_derrotas":[away.iloc[0]["derrotas"]],
        "away_elo":[away.iloc[0]["elo"]],

        "home_ataque_pymc":[home_pymc.iloc[0]["ataque_pymc"]],
        "home_defensa_pymc":[home_pymc.iloc[0]["defensa_pymc"]],
        "away_ataque_pymc":[away_pymc.iloc[0]["ataque_pymc"]],
        "away_defensa_pymc":[away_pymc.iloc[0]["defensa_pymc"]]

    })

    return fila

# =====================================================
# FUNCION SIMULAR
# =====================================================

def simular(
    local,
    visitante
):

    fila = crear_fila(
        local,
        visitante
    )

    lambda_local = max(
        0.05,
        modelo_home.predict(
            fila
        )[0]
    )

    lambda_visitante = max(
        0.05,
        modelo_away.predict(
            fila
        )[0]
    )

    goles_local = np.random.poisson(
        lambda_local,
        N_SIMULACIONES
    )

    goles_visitante = np.random.poisson(
        lambda_visitante,
        N_SIMULACIONES
    )

    p_local = np.mean(
        goles_local > goles_visitante
    )

    p_empate = np.mean(
        goles_local == goles_visitante
    )

    p_visitante = np.mean(
        goles_local < goles_visitante
    )

    return (
        lambda_local,
        lambda_visitante,
        p_local,
        p_empate,
        p_visitante,
        goles_local,
        goles_visitante
    )

# =====================================================
# A VS B
# =====================================================

(
    lambda_a,
    lambda_b,
    p_a,
    p_emp,
    p_b,
    _,
    _
) = simular(
    EQUIPO_A,
    EQUIPO_B
)

# =====================================================
# B VS A
# =====================================================

(
    lambda_b2,
    lambda_a2,
    p_b2,
    p_emp2,
    p_a2,
    _,
    _
) = simular(
    EQUIPO_B,
    EQUIPO_A
)

# =====================================================
# PROMEDIO NEUTRAL
# =====================================================

lambda_a = (
    lambda_a +
    lambda_a2
) / 2

lambda_b = (
    lambda_b +
    lambda_b2
) / 2

p_a = (
    p_a +
    p_a2
) / 2

p_b = (
    p_b +
    p_b2
) / 2

p_emp = (
    p_emp +
    p_emp2
) / 2

# =====================================================
# MONTE CARLO FINAL
# =====================================================

goles_a = np.random.poisson(
    lambda_a,
    N_SIMULACIONES
)

goles_b = np.random.poisson(
    lambda_b,
    N_SIMULACIONES
)

marcadores = pd.DataFrame({

    "local":
        goles_a,

    "visitante":
        goles_b

})

marcadores["marcador"] = (

    marcadores["local"]
    .astype(str)

    +

    "-"

    +

    marcadores["visitante"]
    .astype(str)

)

top = (

    marcadores["marcador"]

    .value_counts(
        normalize=True
    )

    .reset_index()

)

top.columns = [
    "marcador",
    "probabilidad"
]

# =====================================================
# HEATMAP
# =====================================================

matriz = np.zeros(
    (
        MAX_GOLES + 1,
        MAX_GOLES + 1
    )
)

for _, fila in top.iterrows():

    try:

        g1, g2 = fila[
            "marcador"
        ].split("-")

        g1 = int(g1)
        g2 = int(g2)

        if (
            g1 <= MAX_GOLES
            and
            g2 <= MAX_GOLES
        ):

            matriz[
                g1,
                g2
            ] = fila[
                "probabilidad"
            ]

    except:
        pass

# =====================================================
# DASHBOARD
# =====================================================

fig = plt.figure(
    figsize=(14, 9)
)

fig.suptitle(
    f"{EQUIPO_A} vs {EQUIPO_B}",
    fontsize=20,
    fontweight="bold"
)

# ---------- PROBABILIDADES ----------

ax1 = plt.subplot2grid(
    (2, 2),
    (0, 0)
)

ax1.axis("off")

texto = f"""
PROBABILIDADES

{EQUIPO_A}: {p_a*100:.2f}%

Empate: {p_emp*100:.2f}%

{EQUIPO_B}: {p_b*100:.2f}%

GOLES ESPERADOS

{EQUIPO_A}: {lambda_a:.2f}

{EQUIPO_B}: {lambda_b:.2f}
"""

ax1.text(
    0,
    1,
    texto,
    fontsize=12,
    va="top"
)

# ---------- TOP MARCADORES ----------

ax2 = plt.subplot2grid(
    (2, 2),
    (0, 1)
)

ax2.axis("off")

texto_top = "TOP 10 MARCADORES\n\n"

for _, fila in top.head(10).iterrows():

    texto_top += (
        f"{fila['marcador']}    "
        f"{fila['probabilidad']*100:.2f}%\n"
    )

ax2.text(
    0,
    1,
    texto_top,
    fontsize=12,
    va="top"
)

# ---------- HEATMAP ----------

ax3 = plt.subplot2grid(
    (2, 2),
    (1, 0),
    colspan=2
)

im = ax3.imshow(
    matriz
)

ax3.set_title(
    "Heatmap de Marcadores"
)

ax3.set_xlabel(
    EQUIPO_B
)

ax3.set_ylabel(
    EQUIPO_A
)

plt.colorbar(
    im,
    ax=ax3
)

plt.tight_layout()

plt.savefig(
    "dashboard_partido.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nDashboard generado:")
print("dashboard_partido.png")

plt.show()