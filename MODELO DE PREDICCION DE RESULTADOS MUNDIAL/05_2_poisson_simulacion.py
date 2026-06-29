import pandas as pd
import numpy as np
import pickle

from scipy.stats import poisson

# =====================================================
# CONFIGURACION
# =====================================================

# =====================================================
# SELECCION DE EQUIPOS
# =====================================================

print("\n==========================")
print("PREDICTOR DE PARTIDOS")
print("==========================")

LOCAL = input(
    "\nEquipo local: "
).strip()

VISITANTE = input(
    "Equipo visitante: "
).strip() 

MAX_GOLES = 6

# =====================================================
# PARAMETRO DIXON COLES
# =====================================================

RHO = -0.08

# =====================================================
# FUNCION DIXON COLES
# =====================================================

def tau_dc(
    x,
    y,
    lambda_home,
    lambda_away,
    rho
):

    if x == 0 and y == 0:

        return (
            1
            -
            lambda_home
            *
            lambda_away
            *
            rho
        )

    elif x == 0 and y == 1:

        return (
            1
            +
            lambda_home
            *
            rho
        )

    elif x == 1 and y == 0:

        return (
            1
            +
            lambda_away
            *
            rho
        )

    elif x == 1 and y == 1:

        return (
            1
            -
            rho
        )

    else:

        return 1

# =====================================================
# CARGAR MODELOS
# =====================================================

print("Cargando modelos...")

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
# OBTENER EQUIPOS
# =====================================================

home = features[
    features["equipo"] == LOCAL
]

away = features[
    features["equipo"] == VISITANTE
]

if len(home) == 0:

    raise Exception(
        f"No existe {LOCAL}"
    )

if len(away) == 0:

    raise Exception(
        f"No existe {VISITANTE}"
    )

home_pymc = ratings_pymc[
    ratings_pymc["equipo"] == LOCAL
]

away_pymc = ratings_pymc[
    ratings_pymc["equipo"] == VISITANTE
]

# =====================================================
# CREAR FILA PARA XGBOOST
# =====================================================

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

# =====================================================
# GOLES ESPERADOS
# =====================================================

lambda_home = modelo_home.predict(
    fila
)[0]

lambda_away = modelo_away.predict(
    fila
)[0]

lambda_home = max(
    0.05,
    lambda_home
)

lambda_away = max(
    0.05,
    lambda_away
)

print("\n==========================")
print("GOLES ESPERADOS")
print("==========================")

print(
    f"{LOCAL}: {lambda_home:.3f}"
)

print(
    f"{VISITANTE}: {lambda_away:.3f}"
)

# =====================================================
# MATRIZ POISSON + DIXON COLES
# =====================================================

matriz = np.zeros(
    (
        MAX_GOLES + 1,
        MAX_GOLES + 1
    )
)

for goles_local in range(
    MAX_GOLES + 1
):

    for goles_visitante in range(
        MAX_GOLES + 1
    ):

        p_home = poisson.pmf(
            goles_local,
            lambda_home
        )

        p_away = poisson.pmf(
            goles_visitante,
            lambda_away
        )

        ajuste = tau_dc(

            goles_local,

            goles_visitante,

            lambda_home,

            lambda_away,

            RHO

        )

        matriz[
            goles_local,
            goles_visitante
        ] = (

            p_home
            *
            p_away
            *
            ajuste

        )

# =====================================================
# NORMALIZAR
# =====================================================

matriz = matriz / matriz.sum()

# =====================================================
# PROBABILIDADES
# =====================================================

p_local = 0
p_empate = 0
p_visitante = 0

for i in range(
    MAX_GOLES + 1
):

    for j in range(
        MAX_GOLES + 1
    ):

        if i > j:

            p_local += matriz[i, j]

        elif i == j:

            p_empate += matriz[i, j]

        else:

            p_visitante += matriz[i, j]

# =====================================================
# TOP MARCADORES
# =====================================================

resultados = []

for i in range(
    MAX_GOLES + 1
):

    for j in range(
        MAX_GOLES + 1
    ):

        resultados.append({

            "marcador":
                f"{i}-{j}",

            "probabilidad":
                matriz[i, j]

        })

resultados = pd.DataFrame(
    resultados
)

resultados = resultados.sort_values(
    "probabilidad",
    ascending=False
)

# =====================================================
# RESULTADOS
# =====================================================

print("\n==========================")
print("PROBABILIDADES")
print("==========================")

print(
    f"{LOCAL} gana: {p_local*100:.2f}%"
)

print(
    f"Empate: {p_empate*100:.2f}%"
)

print(
    f"{VISITANTE} gana: {p_visitante*100:.2f}%"
)

print("\n==========================")
print("TOP 10 MARCADORES")
print("==========================")

print(
    resultados
    .head(10)
    .to_string(index=False)
)

# =====================================================
# GUARDAR
# =====================================================

resultados.to_csv(
    "probabilidades_partido.csv",
    index=False
)

print("\nArchivo generado:")
print("probabilidades_partido.csv")

# =====================================================
# MOSTRAR EQUIPOS DISPONIBLES
# =====================================================

print("\nEquipos disponibles:\n")

for equipo in sorted(
    features["equipo"].tolist()
):
    print(equipo)