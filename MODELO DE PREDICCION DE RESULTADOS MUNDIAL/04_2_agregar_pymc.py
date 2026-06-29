import pandas as pd

print("Cargando datasets...")

df = pd.read_csv(
    "dataset_xgboost.csv"
)

partidos = pd.read_csv(
    "dataset_mundial_2026.csv"
)

pymc = pd.read_csv(
    "ratings_pymc.csv"
)

# =====================================
# HOME PYMC
# =====================================

home_pymc = pymc.copy()

home_pymc.columns = [
    "home_team",
    "home_ataque_pymc",
    "home_defensa_pymc"
]

# =====================================
# AWAY PYMC
# =====================================

away_pymc = pymc.copy()

away_pymc.columns = [
    "away_team",
    "away_ataque_pymc",
    "away_defensa_pymc"
]

# =====================================
# RECUPERAR EQUIPOS
# =====================================

equipos = partidos[
    ["home_team", "away_team"]
].copy()

df = pd.concat(
    [equipos, df],
    axis=1
)

# =====================================
# MERGE HOME
# =====================================

df = df.merge(
    home_pymc,
    on="home_team",
    how="left"
)

# =====================================
# MERGE AWAY
# =====================================

df = df.merge(
    away_pymc,
    on="away_team",
    how="left"
)

# =====================================
# ELIMINAR NOMBRES
# =====================================

df.drop(
    columns=[
        "home_team",
        "away_team"
    ],
    inplace=True
)

# =====================================
# GUARDAR
# =====================================

df.to_csv(
    "dataset_xgboost_final.csv",
    index=False
)

print("\nArchivo generado")

print(
    "dataset_xgboost_final.csv"
)

print(
    f"\nColumnas: {len(df.columns)}"
)

print(
    df.columns.tolist()
)
