import pandas as pd
import numpy as np
import pickle

# =====================================================
# CONFIGURACION
# =====================================================

LOCAL = "Mexico"
VISITANTE = "Argentina"

# =====================================================
# CARGAR ARCHIVOS
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
# OBTENER FEATURES EQUIPO
# =====================================================

home = features[
    features["equipo"] == LOCAL
].copy()

away = features[
    features["equipo"] == VISITANTE
].copy()

if len(home) == 0:

    raise Exception(
        f"No existe {LOCAL}"
    )

if len(away) == 0:

    raise Exception(
        f"No existe {VISITANTE}"
    )

# =====================================================
# PYMC
# =====================================================

home_pymc = ratings_pymc[
    ratings_pymc["equipo"] == LOCAL
].copy()

away_pymc = ratings_pymc[
    ratings_pymc["equipo"] == VISITANTE
].copy()

# =====================================================
# CREAR FILA
# =====================================================

fila = pd.DataFrame({

    "neutral":[1],

    "dias":[0],

    "peso_tiempo":[1],

    "peso_torneo":[1],

    "peso_total":[1],

    "home_ataque":[
        home.iloc[0]["ataque"]
    ],

    "home_defensa":[
        home.iloc[0]["defensa"]
    ],

    "home_diferencia_goles":[
        home.iloc[0]["diferencia_goles"]
    ],

    "home_forma_5":[
        home.iloc[0]["forma_5"]
    ],

    "home_forma_10":[
        home.iloc[0]["forma_10"]
    ],

    "home_forma_30":[
        home.iloc[0]["forma_30"]
    ],

    "home_puntos_ponderados":[
        home.iloc[0]["puntos_ponderados"]
    ],

    "home_victorias":[
        home.iloc[0]["victorias"]
    ],

    "home_empates":[
        home.iloc[0]["empates"]
    ],

    "home_derrotas":[
        home.iloc[0]["derrotas"]
    ],

    "home_elo":[
        home.iloc[0]["elo"]
    ],

    "away_ataque":[
        away.iloc[0]["ataque"]
    ],

    "away_defensa":[
        away.iloc[0]["defensa"]
    ],

    "away_diferencia_goles":[
        away.iloc[0]["diferencia_goles"]
    ],

    "away_forma_5":[
        away.iloc[0]["forma_5"]
    ],

    "away_forma_10":[
        away.iloc[0]["forma_10"]
    ],

    "away_forma_30":[
        away.iloc[0]["forma_30"]
    ],

    "away_puntos_ponderados":[
        away.iloc[0]["puntos_ponderados"]
    ],

    "away_victorias":[
        away.iloc[0]["victorias"]
    ],

    "away_empates":[
        away.iloc[0]["empates"]
    ],

    "away_derrotas":[
        away.iloc[0]["derrotas"]
    ],

    "away_elo":[
        away.iloc[0]["elo"]
    ],

    "home_ataque_pymc":[
        home_pymc.iloc[0]["ataque_pymc"]
    ],

    "home_defensa_pymc":[
        home_pymc.iloc[0]["defensa_pymc"]
    ],

    "away_ataque_pymc":[
        away_pymc.iloc[0]["ataque_pymc"]
    ],

    "away_defensa_pymc":[
        away_pymc.iloc[0]["defensa_pymc"]
    ]

})

# =====================================================
# PREDICCION XGBOOST
# =====================================================

lambda_home = modelo_home.predict(
    fila
)[0]

lambda_away = modelo_away.predict(
    fila
)[0]

# =====================================================
# AJUSTE SEGURIDAD
# =====================================================

lambda_home = max(
    0.05,
    lambda_home
)

lambda_away = max(
    0.05,
    lambda_away
)

# =====================================================
# RESULTADOS
# =====================================================

print("\n========================")
print("PREDICCION")
print("========================")

print(
    f"{LOCAL}: {lambda_home:.3f}"
)

print(
    f"{VISITANTE}: {lambda_away:.3f}"
)

print("\nInterpretacion:")

print(
    f"Goles esperados {LOCAL}: {lambda_home:.2f}"
)

print(
    f"Goles esperados {VISITANTE}: {lambda_away:.2f}"
)