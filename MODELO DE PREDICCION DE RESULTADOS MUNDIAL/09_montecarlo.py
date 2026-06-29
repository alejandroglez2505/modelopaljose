import pandas as pd
import numpy as np
import pickle

# =====================================================
# CONFIGURACION
# =====================================================

LOCAL = "Mexico"
VISITANTE = "Argentina"

N_SIMULACIONES = 100000

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
# FILA MODELO
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

lambda_home = max(
    0.05,
    modelo_home.predict(
        fila
    )[0]
)

lambda_away = max(
    0.05,
    modelo_away.predict(
        fila
    )[0]
)

print("\n========================")
print("GOLES ESPERADOS")
print("========================")

print(
    f"{LOCAL}: {lambda_home:.3f}"
)

print(
    f"{VISITANTE}: {lambda_away:.3f}"
)

# =====================================================
# MONTE CARLO
# =====================================================

print(
    f"\nSimulando {N_SIMULACIONES:,} partidos..."
)

goles_local = np.random.poisson(
    lambda_home,
    N_SIMULACIONES
)

goles_visitante = np.random.poisson(
    lambda_away,
    N_SIMULACIONES
)

# =====================================================
# RESULTADOS
# =====================================================

victorias_local = np.sum(
    goles_local > goles_visitante
)

empates = np.sum(
    goles_local == goles_visitante
)

victorias_visitante = np.sum(
    goles_local < goles_visitante
)

# =====================================================
# PROBABILIDADES
# =====================================================

p_local = (
    victorias_local
    / N_SIMULACIONES
)

p_empate = (
    empates
    / N_SIMULACIONES
)

p_visitante = (
    victorias_visitante
    / N_SIMULACIONES
)

# =====================================================
# MARCADORES
# =====================================================

marcadores = pd.DataFrame({

    "local":
        goles_local,

    "visitante":
        goles_visitante

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

top_marcadores = (

    marcadores["marcador"]

    .value_counts(
        normalize=True
    )

    .reset_index()

)

top_marcadores.columns = [

    "marcador",
    "probabilidad"

]

# =====================================================
# RESULTADOS
# =====================================================

print("\n========================")
print("MONTE CARLO")
print("========================")

print(
    f"{LOCAL} gana: {p_local*100:.2f}%"
)

print(
    f"Empate: {p_empate*100:.2f}%"
)

print(
    f"{VISITANTE} gana: {p_visitante*100:.2f}%"
)

print("\n========================")
print("TOP 15 MARCADORES")
print("========================")

print(
    top_marcadores
    .head(15)
    .to_string(index=False)
)

# =====================================================
# GUARDAR
# =====================================================

top_marcadores.to_csv(
    "montecarlo_resultados.csv",
    index=False
)

print("\nArchivo generado:")
print(
    "montecarlo_resultados.csv"
)