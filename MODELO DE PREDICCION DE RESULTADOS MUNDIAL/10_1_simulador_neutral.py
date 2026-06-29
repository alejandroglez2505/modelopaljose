import pandas as pd
import numpy as np
import pickle

# =====================================================
# CONFIG
# =====================================================

N_SIMULACIONES = 100000

# =====================================================
# CARGAR DATOS
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

local_idx = int(
    input(
        "\nNumero primer equipo: "
    )
)

visitante_idx = int(
    input(
        "Numero segundo equipo: "
    )
)

EQUIPO_A = equipos[
    local_idx - 1
]

EQUIPO_B = equipos[
    visitante_idx - 1
]

print("\n========================")
print("PARTIDO NEUTRAL")
print("========================")

print(
    f"{EQUIPO_A} vs {EQUIPO_B}"
)

# =====================================================
# FUNCION
# =====================================================

def obtener_fila(
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
# SIMULACION
# =====================================================

def simular(
    local,
    visitante
):

    fila = obtener_fila(
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
        p_visitante
    )

# =====================================================
# A VS B
# =====================================================

(
    lambda_a,
    lambda_b,
    p_a,
    p_emp,
    p_b
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
    p_a2
) = simular(
    EQUIPO_B,
    EQUIPO_A
)

# =====================================================
# PROMEDIOS
# =====================================================

prob_a = (
    p_a + p_a2
) / 2

prob_b = (
    p_b + p_b2
) / 2

prob_empate = (
    p_emp + p_emp2
) / 2

goles_a = (
    lambda_a + lambda_a2
) / 2

goles_b = (
    lambda_b + lambda_b2
) / 2

# =====================================================
# RESULTADOS
# =====================================================

print("\n========================")
print("RESULTADO NEUTRAL")
print("========================")

print(
    f"{EQUIPO_A}: {goles_a:.3f} goles esperados"
)

print(
    f"{EQUIPO_B}: {goles_b:.3f} goles esperados"
)

print("\n========================")
print("PROBABILIDADES")
print("========================")

print(
    f"{EQUIPO_A} gana: {prob_a*100:.2f}%"
)

print(
    f"Empate: {prob_empate*100:.2f}%"
)

print(
    f"{EQUIPO_B} gana: {prob_b*100:.2f}%"
)

print(
    f"\nSuma: {(prob_a + prob_empate + prob_b)*100:.2f}%"
)